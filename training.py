import argparse
import json
import time
import warnings
from pathlib import Path

import config
from ppo import PPO
import tensorflow as tf
from tqdm import tqdm


def format_float(value):
    return f"{value:.3f}"


def summarize_epoch(stats, elapsed_seconds, num_timesteps):
    rewards = [item["reward"] for item in stats]
    lengths = [item["length"] for item in stats]
    pipes = [item["pipes"] for item in stats]
    terminated = [item["terminated"] for item in stats]

    return {
        "timesteps": num_timesteps,
        "avg_reward": sum(rewards) / len(rewards),
        "max_reward": max(rewards),
        "avg_length": sum(lengths) / len(lengths),
        "max_length": max(lengths),
        "avg_pipes": sum(pipes) / len(pipes),
        "terminated": sum(1 for item in terminated if item),
        "seconds": elapsed_seconds,
    }


def format_validation(prefix, stats):
    return (
        f"{prefix}: "
        f"avg_score={stats['avg_score']:.2f} "
        f"max_score={stats['max_score']} "
        f"avg_steps={stats['avg_steps']:.1f} "
        f"max_steps={stats['max_steps']} "
        f"target_hits={stats['target_hits']}/{stats['episodes']} "
        f"scores={stats['scores']}"
    )


def entropy_coefficient_for_epoch(epoch):
    start = config.PPO_ENTROPY_COEFFICIENT_START
    end = config.PPO_ENTROPY_COEFFICIENT_END
    decay_epochs = max(1, config.PPO_ENTROPY_DECAY_EPOCHS)
    progress = min(1.0, max(0.0, (epoch - 1) / decay_epochs))
    return start + ((end - start) * progress)


def create_checkpoint_managers(ppo, checkpoint_dir, max_to_keep):
    checkpoint = tf.train.Checkpoint(
        epoch=tf.Variable(0, dtype=tf.int64, trainable=False),
        actor=ppo.actor,
        critic=ppo.critic,
        actor_optimizer=ppo.actor.optimizer,
        critic_optimizer=ppo.critic.optimizer,
        entropy_coefficient=ppo.entropy_coefficient,
    )
    root = Path(checkpoint_dir)
    return checkpoint, {
        "latest": tf.train.CheckpointManager(
            checkpoint,
            str(root / "latest"),
            max_to_keep=max_to_keep,
        ),
        "best": tf.train.CheckpointManager(
            checkpoint,
            str(root / "best"),
            max_to_keep=max_to_keep,
        ),
    }


def restore_checkpoint(ppo, checkpoint, checkpoint_path):
    path = Path(checkpoint_path)
    checkpoint_path = tf.train.latest_checkpoint(str(path)) if path.is_dir() else str(path)
    if not checkpoint_path or not Path(f"{checkpoint_path}.index").exists():
        raise ValueError(f"Checkpoint not found: {checkpoint_path}")

    state = tf.zeros(
        (
            1,
            config.OBS_HEIGHT,
            config.OBS_WIDTH,
            config.OBS_CHANNELS * config.FRAME_STACK,
        ),
        dtype=tf.uint8,
    )
    ppo.actor(state)
    ppo.critic(state)
    ppo.actor.optimizer.build(ppo.actor.trainable_variables)
    ppo.critic.optimizer.build(ppo.critic.trainable_variables)

    status = checkpoint.restore(checkpoint_path)
    status.assert_existing_objects_matched()
    restored_epoch = int(checkpoint.epoch.numpy())
    tqdm.write(f"resumed checkpoint: {checkpoint_path} epoch={restored_epoch}")
    return restored_epoch


def save_checkpoint(checkpoint, manager, epoch, label, validation_stats=None):
    checkpoint.epoch.assign(epoch)
    path = manager.save(checkpoint_number=epoch)
    message = f"checkpoint[{label}]: {path} epoch={epoch}"
    if validation_stats is not None:
        message += (
            f" avg_score={validation_stats['avg_score']:.2f}"
            f" max_score={validation_stats['max_score']}"
        )
        metadata_path = Path(manager.directory) / "best_validation.json"
        metadata_path.write_text(
            json.dumps(
                {
                    "epoch": epoch,
                    "avg_score": validation_stats["avg_score"],
                    "max_score": validation_stats["max_score"],
                    "scores": validation_stats["scores"],
                },
                indent=2,
            )
            + "\n",
            encoding="ascii",
        )
    tqdm.write(message)
    return path


def load_best_validation(checkpoint_dir):
    metadata_path = Path(checkpoint_dir) / "best" / "best_validation.json"
    if not metadata_path.exists():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="ascii"))
        return metadata["avg_score"], metadata["max_score"]
    except (OSError, ValueError, KeyError, TypeError):
        warnings.warn(
            f"Ignoring unreadable best-checkpoint metadata at {metadata_path}.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None


def main():
    parser = argparse.ArgumentParser(description="Train the Flappy Bird PPO agent.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--debug-window", action="store_true")
    parser.add_argument("--show-game-window", action="store_true")
    parser.add_argument("--env-backend", choices=["cpp_vector", "fast", "subprocess"], default=None)
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--ppo-epochs", type=int, default=None)
    parser.add_argument("--minibatch-size", type=int, default=None)
    parser.add_argument("--target-kl", type=float, default=None)
    parser.add_argument("--entropy-start", type=float, default=None)
    parser.add_argument("--entropy-end", type=float, default=None)
    parser.add_argument("--entropy-decay-epochs", type=int, default=None)
    parser.add_argument("--validate-every", type=int, default=0)
    parser.add_argument("--validation-episodes", type=int, default=None)
    parser.add_argument("--validation-max-steps", type=int, default=None)
    parser.add_argument("--validation-target-score", type=int, default=None)
    parser.add_argument("--validation-backend", choices=["cpp_vector", "fast", "subprocess"], default="cpp_vector")
    parser.add_argument("--validate-sim-mismatch", action="store_true")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--checkpoint-keep", type=int, default=3)
    parser.add_argument("--no-checkpoints", action="store_true")
    parser.add_argument("--resume-from", default=None)
    args = parser.parse_args()

    if args.rollout_steps is not None:
        config.PPO_ROLLOUT_STEPS = args.rollout_steps
    if args.num_envs is not None:
        config.NUM_ENVS = args.num_envs
    if args.max_steps is not None:
        config.MAX_NUM_STEPS = args.max_steps
    if args.ppo_epochs is not None:
        config.PPO_EPOCHS = args.ppo_epochs
    if args.minibatch_size is not None:
        config.PPO_MINIBATCH_SIZE = args.minibatch_size
    if args.target_kl is not None:
        config.PPO_TARGET_KL = args.target_kl
    if args.entropy_start is not None:
        config.PPO_ENTROPY_COEFFICIENT_START = args.entropy_start
    if args.entropy_end is not None:
        config.PPO_ENTROPY_COEFFICIENT_END = args.entropy_end
    if args.entropy_decay_epochs is not None:
        config.PPO_ENTROPY_DECAY_EPOCHS = args.entropy_decay_epochs

    if config.MAX_NUM_STEPS < 128:
        warnings.warn(
            "MAX_NUM_STEPS is below the first-pipe learning horizon. "
            "This is fine for smoke tests, but real training should use a much larger cap.",
            RuntimeWarning,
            stacklevel=2,
        )

    ppo = PPO(
        debug_window=args.debug_window,
        show_game_window=args.show_game_window,
        env_backend=args.env_backend,
    )
    checkpoint, checkpoint_managers = create_checkpoint_managers(
        ppo,
        args.checkpoint_dir,
        max(1, args.checkpoint_keep),
    )
    restored_epoch = 0
    if args.resume_from is not None:
        restored_epoch = restore_checkpoint(ppo, checkpoint, args.resume_from)
    if args.no_checkpoints:
        checkpoint = None
        checkpoint_managers = None
    best_validation = None if args.no_checkpoints else load_best_validation(args.checkpoint_dir)
    last_latest_checkpoint_epoch = restored_epoch
    completed_epoch = restored_epoch

    try:
        with tqdm(
            range(restored_epoch + 1, restored_epoch + args.epochs + 1),
            desc="training",
            unit="epoch",
        ) as progress:
            for epoch in progress:
                entropy_coefficient = entropy_coefficient_for_epoch(epoch)
                ppo.set_entropy_coefficient(entropy_coefficient)
                started = time.perf_counter()
                processed_timesteps = ppo.training_epoch()
                elapsed = time.perf_counter() - started
                completed_epoch = epoch

                summary = summarize_epoch(
                    ppo.last_trajectory_stats,
                    elapsed,
                    len(processed_timesteps),
                )
                metrics = ppo.last_training_metrics

                progress.set_postfix({
                    "steps": summary["timesteps"],
                    "avg_reward": format_float(summary["avg_reward"]),
                    "max_reward": format_float(summary["max_reward"]),
                    "avg_len": format_float(summary["avg_length"]),
                    "max_len": summary["max_length"],
                    "avg_pipes": format_float(summary["avg_pipes"]),
                    "done": summary["terminated"],
                    "ent_coef": format_float(entropy_coefficient),
                    "ent": format_float(metrics["actor_entropy"]),
                    "kl": format_float(metrics["approx_kl"]),
                    "clip": format_float(metrics["clip_fraction"]),
                    "ev": format_float(metrics["explained_variance"]),
                    "passes": metrics["ppo_passes"],
                    "kl_stop": metrics["early_stop"],
                    "sec": format_float(summary["seconds"]),
                })

                tqdm.write(
                    "epoch "
                    f"{epoch}: steps={summary['timesteps']} "
                    f"avg_reward={summary['avg_reward']:.3f} "
                    f"max_reward={summary['max_reward']:.3f} "
                    f"avg_len={summary['avg_length']:.1f} "
                    f"max_len={summary['max_length']} "
                    f"avg_pipes={summary['avg_pipes']:.3f} "
                    f"terminated={summary['terminated']}/{len(ppo.last_trajectory_stats)} "
                    f"entropy_coef={entropy_coefficient:.5f} "
                    f"actor_entropy={metrics['actor_entropy']:.4f} "
                    f"approx_kl={metrics['approx_kl']:.5f} "
                    f"clip_fraction={metrics['clip_fraction']:.3f} "
                    f"ppo_passes={metrics['ppo_passes']} "
                    f"kl_early_stop={metrics['early_stop']} "
                    f"policy_loss={metrics['policy_loss']:.4f} "
                    f"value_loss={metrics['value_loss']:.4f} "
                    f"explained_variance={metrics['explained_variance']:.3f} "
                    f"seconds={summary['seconds']:.2f}"
                )

                if (
                    checkpoint is not None
                    and args.checkpoint_every > 0
                    and epoch % args.checkpoint_every == 0
                ):
                    save_checkpoint(
                        checkpoint,
                        checkpoint_managers["latest"],
                        epoch,
                        "latest",
                    )
                    last_latest_checkpoint_epoch = epoch

                if args.validate_every > 0 and epoch % args.validate_every == 0:
                    deterministic_stats = ppo.evaluate_policy(
                        episodes=args.validation_episodes,
                        max_steps=args.validation_max_steps,
                        env_backend=args.validation_backend,
                        target_score=args.validation_target_score,
                        deterministic=True,
                    )
                    stochastic_stats = ppo.evaluate_policy(
                        episodes=args.validation_episodes,
                        max_steps=args.validation_max_steps,
                        env_backend=args.validation_backend,
                        target_score=args.validation_target_score,
                        deterministic=False,
                    )
                    tqdm.write(format_validation(f"validation[{args.validation_backend}:det]", deterministic_stats))
                    tqdm.write(format_validation(f"validation[{args.validation_backend}:sample]", stochastic_stats))

                    validation_key = (
                        deterministic_stats["avg_score"],
                        deterministic_stats["max_score"],
                    )
                    if checkpoint is not None and (
                        best_validation is None or validation_key > best_validation
                    ):
                        save_checkpoint(
                            checkpoint,
                            checkpoint_managers["best"],
                            epoch,
                            "best",
                            deterministic_stats,
                        )
                        best_validation = validation_key

                    if args.validate_sim_mismatch:
                        mismatch = ppo.validate_sim_mismatch(
                            episodes=args.validation_episodes,
                            max_steps=args.validation_max_steps,
                            target_score=args.validation_target_score,
                        )
                        tqdm.write(format_validation("validation[real]", mismatch["real"]))
                        tqdm.write(format_validation("validation[fast]", mismatch["fast"]))
                        tqdm.write(
                            "sim_mismatch: "
                            f"score_gap={mismatch['score_gap']:.2f} "
                            f"step_gap={mismatch['step_gap']:.1f}"
                        )
    finally:
        if checkpoint is not None and completed_epoch > last_latest_checkpoint_epoch:
            save_checkpoint(
                checkpoint,
                checkpoint_managers["latest"],
                completed_epoch,
                "latest",
            )
        ppo.close()


if __name__ == "__main__":
    main()
