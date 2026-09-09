from types import SimpleNamespace

import config
from ppo import PPO


def result(*, bird_y, gap_center, gap_half_height, terminated=False, has_next_pipe=True, passed_pipe=False):
    return SimpleNamespace(
        bird_y=bird_y,
        next_gap_center_y=gap_center,
        next_gap_half_height=gap_half_height,
        terminated=terminated,
        has_next_pipe=has_next_pipe,
        passed_pipe=passed_pipe,
    )


def main():
    centered = result(bird_y=500.0, gap_center=500.0, gap_half_height=100.0)
    offset = result(bird_y=600.0, gap_center=500.0, gap_half_height=100.0)
    closer = result(bird_y=550.0, gap_center=500.0, gap_half_height=100.0)
    terminal = result(
        bird_y=500.0,
        gap_center=500.0,
        gap_half_height=100.0,
        terminated=True,
        has_next_pipe=False,
    )

    assert PPO.navigation_potential(centered) == 1.0
    assert PPO.navigation_potential(offset) == 0.0
    assert PPO.navigation_potential(terminal) == 0.0

    shaping_gain = PPO.reward_from_transition(offset, closer) - config.REWARD_STD
    assert shaping_gain > 0.0
    terminal_reward = PPO.reward_from_transition(centered, terminal)
    assert terminal_reward < config.REWARD_DIE
    print("reward shaping tests passed")


if __name__ == "__main__":
    main()
