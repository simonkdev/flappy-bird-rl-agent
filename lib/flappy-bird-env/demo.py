from __future__ import annotations

import argparse
import sys
import time
from itertools import count
from pathlib import Path


LIB_ROOT = Path(__file__).resolve().parent
RL_ROOT = LIB_ROOT.parent.parent
sys.path.insert(0, str(LIB_ROOT / "python"))

from flappy_env import FlappyEnv, StepResult  # noqa: E402


def print_result(label: str, result: StepResult, action: int | None = None) -> None:
    pixels = result.observation
    action_text = "reset" if action is None else f"action={action}"
    print(
        f"{label}: {action_text}, shape={result.shape}, dtype={result.dtype}, "
        f"range=({min(pixels)}, {max(pixels)}), reward={result.reward:.2f}, "
        f"done={result.terminated}, score={result.score}, sim={result.simulation_time:.3f}"
    )


def run_episode(env: FlappyEnv, episode: int, seed: int, max_steps: int, real_time: bool) -> None:
    result = env.reset_result(seed=seed)
    print_result(f"episode {episode}", result)

    for step in range(max_steps):
        started = time.monotonic()
        action = 1 if step % 8 == 0 else 0
        result = env.step_result(action)

        if step < 8 or result.passed_pipe or result.terminated:
            print_result(f"episode {episode} step {step:03d}", result, action)

        if result.terminated:
            return

        if real_time:
            step_seconds = env.ticks_per_step * env.dt
            elapsed = time.monotonic() - started
            if elapsed < step_seconds:
                time.sleep(step_seconds - elapsed)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the vendored Flappy Bird environment demo from the RL project root."
    )
    parser.add_argument(
        "--exe",
        default=str(RL_ROOT / "lib/flappy-bird-env/build/flappy_env_server"),
        help="Path to flappy_env_server.",
    )
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--infinite", action="store_true")
    parser.add_argument("--real-time", action="store_true")
    parser.add_argument("--debug-window", action="store_true")
    parser.add_argument("--show-game-window", action="store_true")
    args = parser.parse_args()

    with FlappyEnv(
        args.exe,
        seed=args.seed,
        debug_window=args.debug_window,
        show_game_window=args.show_game_window,
    ) as env:
        print(
            f"connected: server={args.exe}, observation=({env.height}, {env.width}, 1), "
            f"ticks_per_step={env.ticks_per_step}, dt={env.dt}"
        )

        episodes = count(1) if args.infinite else range(1, args.episodes + 1)
        try:
            for episode in episodes:
                run_episode(env, episode, args.seed + episode - 1, args.steps, args.real_time)
        except KeyboardInterrupt:
            print("stopped")


if __name__ == "__main__":
    main()
