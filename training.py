import argparse
import time
import warnings

import config
from ppo import PPO
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
    parser.add_argument("--validate-every", type=int, default=0)
    parser.add_argument("--validation-episodes", type=int, default=None)
    parser.add_argument("--validation-max-steps", type=int, default=None)
    parser.add_argument("--validation-target-score", type=int, default=None)
    parser.add_argument("--validation-backend", choices=["cpp_vector", "fast", "subprocess"], default="cpp_vector")
    parser.add_argument("--validate-sim-mismatch", action="store_true")
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

    try:
        with tqdm(range(1, args.epochs + 1), desc="training", unit="epoch") as progress:
            for epoch in progress:
                started = time.perf_counter()
                processed_timesteps = ppo.training_epoch()
                elapsed = time.perf_counter() - started

                summary = summarize_epoch(
                    ppo.last_trajectory_stats,
                    elapsed,
                    len(processed_timesteps),
                )

                progress.set_postfix({
                    "steps": summary["timesteps"],
                    "avg_reward": format_float(summary["avg_reward"]),
                    "max_reward": format_float(summary["max_reward"]),
                    "avg_len": format_float(summary["avg_length"]),
                    "max_len": summary["max_length"],
                    "avg_pipes": format_float(summary["avg_pipes"]),
                    "done": summary["terminated"],
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
                    f"seconds={summary['seconds']:.2f}"
                )

                if args.validate_every > 0 and epoch % args.validate_every == 0:
                    validation_stats = ppo.evaluate_policy(
                        episodes=args.validation_episodes,
                        max_steps=args.validation_max_steps,
                        env_backend=args.validation_backend,
                        target_score=args.validation_target_score,
                    )
                    tqdm.write(format_validation(f"validation[{args.validation_backend}]", validation_stats))

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
        ppo.close()


if __name__ == "__main__":
    main()
