from types import SimpleNamespace

from src import config
from src.ppo import PPO


def result(*, terminated=False, passed_pipe=False):
    return SimpleNamespace(
        terminated=terminated,
        passed_pipe=passed_pipe,
    )


def main():
    assert PPO.reward_from_transition(result()) == config.REWARD_STD
    assert PPO.reward_from_transition(result(passed_pipe=True)) == config.REWARD_PASSED_PIPE
    assert PPO.reward_from_transition(result(terminated=True)) == config.REWARD_DIE
    print("reward tests passed")


if __name__ == "__main__":
    main()
