"""Play checkpoint policies in the visible C++ Flappy Bird environment."""

import argparse
import time
from itertools import count

from src import config
from src.ppo import PPO
from train.training import create_checkpoint_managers, restore_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(
        description="Watch a saved PPO checkpoint play Flappy Bird. Press Ctrl+C to stop."
    )
    parser.add_argument("checkpoint", help="Checkpoint prefix or directory containing checkpoints.")
    parser.add_argument(
        "--episodes",
        type=int,
        default=0,
        help="Episodes to play; 0 repeats until interrupted (default: 0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.VALIDATION_SEED_START,
        help="Seed for the first episode; each following episode increments it.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10_000,
        help="Maximum agent actions per episode (default: 10000).",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions from the actor instead of taking its highest-logit action.",
    )
    parser.add_argument(
        "--no-real-time",
        action="store_true",
        help="Run as quickly as possible instead of pacing to the simulation rate.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Do not open the game window.",
    )
    return parser.parse_args()


def run_episode(ppo, episode, seed, max_steps, stochastic, real_time):
    env = ppo.envs[0]
    result = env.reset_result(seed=seed)
    ppo.framestack_buffers[0].clear()

    for step in range(1, max_steps + 1):
        state = ppo.framestack(result.observation, step - 1, 0)
        action = ppo.decide(state)[0] if stochastic else ppo.decide_deterministic(state)
        started = time.monotonic()
        result = env.step_result(action)
        if real_time:
            frame_seconds = env.ticks_per_step * env.dt
            remaining = frame_seconds - (time.monotonic() - started)
            if remaining > 0:
                time.sleep(remaining)
        if result.terminated:
            print(f"episode={episode} seed={seed} score={result.score} steps={step}")
            return

    print(f"episode={episode} seed={seed} score={result.score} steps={max_steps} capped=1")


def main():
    args = parse_args()
    if args.episodes < 0:
        raise SystemExit("--episodes must be non-negative")
    if args.max_steps < 1:
        raise SystemExit("--max-steps must be positive")

    ppo = PPO(
        env_backend="subprocess",
        num_envs=1,
        show_game_window=not args.headless,
    )
    checkpoint, _ = create_checkpoint_managers(ppo, "/tmp/flappyrl-watch-checkpoint", 1)
    try:
        epoch = restore_checkpoint(ppo, checkpoint, args.checkpoint)
        mode = "stochastic" if args.stochastic else "deterministic"
        print(f"watching epoch={epoch} mode={mode}; press Ctrl+C to stop")
        episodes = range(1, args.episodes + 1) if args.episodes else count(1)
        for episode in episodes:
            run_episode(
                ppo,
                episode,
                args.seed + episode - 1,
                args.max_steps,
                args.stochastic,
                not args.no_real_time,
            )
    except KeyboardInterrupt:
        print("stopped")
    finally:
        ppo.close()


if __name__ == "__main__":
    main()
