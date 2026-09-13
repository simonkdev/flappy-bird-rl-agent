"""Equivalence test for the isolated smooth-playback stepping prototype."""

import sys
from pathlib import Path

from train.playback_microsteps import microstep_actions, step_policy_action


ENV_PYTHON_DIR = Path(__file__).resolve().parents[1] / "lib/flappy-bird-env/python"
sys.path.insert(0, str(ENV_PYTHON_DIR))

from flappy_env import FlappyEnv  # noqa: E402


def assert_same_boundary(macro, micro) -> None:
    assert macro.observation == micro.result.observation
    assert macro.reward == micro.reward
    assert macro.terminated == micro.result.terminated
    assert macro.alive == micro.result.alive
    assert macro.score == micro.result.score
    assert macro.passed_pipe == micro.passed_pipe
    assert macro.simulation_time == micro.result.simulation_time


def action_patterns():
    yield [0] * 96
    yield [1] * 96
    yield [step % 2 for step in range(96)]
    yield [1 if step % 7 == 0 else 0 for step in range(96)]

    state = 0x1234ABCD
    randomized = []
    for _ in range(96):
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        randomized.append((state >> 31) & 1)
    yield randomized


def main() -> None:
    assert microstep_actions(0, 4) == (0, 0, 0, 0)
    assert microstep_actions(1, 4) == (1, 0, 0, 0)

    executable = Path(__file__).resolve().parents[1] / "lib/flappy-bird-env/build/flappy_env_server"
    for seed in (0, 1, 99, 12345, 3_000_000_000):
        for actions in action_patterns():
            with FlappyEnv(executable, ticks_per_step=4) as macro_env, FlappyEnv(
                executable,
                ticks_per_step=1,
            ) as micro_env:
                macro_reset = macro_env.reset_result(seed=seed)
                micro_reset = micro_env.reset_result(seed=seed)
                assert macro_reset == micro_reset

                for action in actions:
                    macro = macro_env.step_result(action)
                    micro = step_policy_action(micro_env, action, 4)
                    assert_same_boundary(macro, micro)
                    if macro.terminated:
                        break

    print("playback microstep equivalence tests passed")


if __name__ == "__main__":
    main()
