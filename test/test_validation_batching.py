from src.ppo import PPO


class FakePPO:
    _summarize_evaluation = staticmethod(PPO._summarize_evaluation)

    def __init__(self):
        self.calls = []

    def _evaluate_policy_batch(self, **kwargs):
        self.calls.append(kwargs)
        seed_start = kwargs["seed_start"]
        episodes = kwargs["episodes"]
        return {
            "scores": [seed_start + offset for offset in range(episodes)],
            "steps": [10 + offset for offset in range(episodes)],
        }


def main():
    ppo = FakePPO()
    stats = PPO.evaluate_policy(
        ppo,
        episodes=5,
        max_steps=100,
        env_backend="cpp_vector",
        deterministic=True,
        target_score=500,
        seed_start=100,
        batch_size=2,
    )

    assert [call["episodes"] for call in ppo.calls] == [2, 2, 1]
    assert [call["seed_start"] for call in ppo.calls] == [100, 102, 104]
    assert stats["scores"] == [100, 101, 102, 103, 104]
    assert stats["steps"] == [10, 11, 10, 11, 10]
    assert stats["episodes"] == 5
    print("validation batching tests passed")


if __name__ == "__main__":
    main()
