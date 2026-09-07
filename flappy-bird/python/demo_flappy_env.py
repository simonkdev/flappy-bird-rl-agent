from __future__ import annotations

import argparse
import time
from itertools import count
from pathlib import Path

from flappy_env import FlappyEnv, StepResult


def describe(prefix: str, result: StepResult, action: int | None = None) -> None:
    pixels = result.observation
    min_value = min(pixels) if pixels else None
    max_value = max(pixels) if pixels else None
    action_text = "reset" if action is None else f"action={action}"
    print(
        f"{prefix}: {action_text}, shape={result.shape}, dtype={result.dtype}, "
        f"range=({min_value}, {max_value}), reward={result.reward:.2f}, "
        f"terminated={result.terminated}, alive={result.alive}, score={result.score}, "
        f"passed_pipe={result.passed_pipe}, sim_time={result.simulation_time:.3f}"
    )


def run_scripted_episode(
    env: FlappyEnv,
    episode: int,
    max_steps: int,
    seed: int,
    *,
    quiet: bool = False,
    real_time: bool = False,
) -> list[bytes]:
    frames: list[bytes] = []
    result = env.reset_result(seed=seed)
    frames.append(result.observation)
    if not quiet:
        describe(f"episode {episode}", result)

    for step in range(max_steps):
        action = 1 if step % 8 == 0 else 0
        start = time.monotonic()
        result = env.step_result(action)
        frames.append(result.observation)

        if not quiet and (step < 8 or result.terminated or result.passed_pipe):
            describe(f"episode {episode} step {step:03d}", result, action)

        if result.terminated:
            break

        if real_time:
            target_seconds = env.ticks_per_step * env.dt
            elapsed = time.monotonic() - start
            if elapsed < target_seconds:
                time.sleep(target_seconds - elapsed)

    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Demonstrate the Flappy Bird C++ environment interface.")
    parser.add_argument("--exe", default="build_check/flappy_env_server", help="Path to flappy_env_server.")
    parser.add_argument("--width", type=int, default=42)
    parser.add_argument("--height", type=int, default=42)
    parser.add_argument("--ticks", type=int, default=4)
    parser.add_argument("--dt", type=float, default=1.0 / 60.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--episodes", type=int, default=2, help="Number of episodes to run.")
    parser.add_argument("--infinite", action="store_true", help="Run episodes until interrupted with Ctrl+C.")
    parser.add_argument("--real-time", action="store_true", help="Pace demo stepping to simulated real time.")
    parser.add_argument("--debug-window", action="store_true")
    parser.add_argument("--show-game-window", action="store_true")
    args = parser.parse_args()

    executable = Path(args.exe)
    with FlappyEnv(
        executable,
        width=args.width,
        height=args.height,
        ticks_per_step=args.ticks,
        dt=args.dt,
        seed=args.seed,
        debug_window=args.debug_window,
        show_game_window=args.show_game_window,
    ) as env:
        print(
            f"connected: observation=({env.height}, {env.width}, 1), "
            f"ticks_per_step={env.ticks_per_step}, dt={env.dt}"
        )

        try:
            env.step_result(99)
        except ValueError as exc:
            print(f"invalid action check: {exc}")

        if args.infinite:
            print("running infinite scripted episodes; press Ctrl+C to stop")
            try:
                for episode in count(1):
                    run_scripted_episode(
                        env,
                        episode,
                        args.steps,
                        args.seed + episode - 1,
                        real_time=args.real_time,
                    )
            except KeyboardInterrupt:
                print("stopped")
        else:
            runs = [
                run_scripted_episode(
                    env,
                    episode,
                    args.steps,
                    args.seed + episode - 1,
                    real_time=args.real_time,
                )
                for episode in range(1, args.episodes + 1)
            ]

            if args.episodes >= 2:
                with FlappyEnv(
                    executable,
                    width=args.width,
                    height=args.height,
                    ticks_per_step=args.ticks,
                    dt=args.dt,
                    seed=args.seed,
                ) as replay_env:
                    replay = run_scripted_episode(replay_env, 1, args.steps, args.seed, quiet=True)
                deterministic = runs[0] == replay
                print(f"deterministic same-seed scripted replay: {deterministic}")


if __name__ == "__main__":
    main()
