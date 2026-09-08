import argparse
import time

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


def main():
    parser = argparse.ArgumentParser(description="Train the Flappy Bird PPO agent.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--debug-window", action="store_true")
    parser.add_argument("--show-game-window", action="store_true")
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--ppo-epochs", type=int, default=None)
    parser.add_argument("--minibatch-size", type=int, default=None)
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

    ppo = PPO(
        debug_window=args.debug_window,
        show_game_window=args.show_game_window,
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
    finally:
        ppo.close()


if __name__ == "__main__":
    main()
