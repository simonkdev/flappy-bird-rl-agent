from pathlib import Path

from vector_flappy_env import VectorFlappyEnv


def main() -> None:
    executable = Path(__file__).resolve().parents[1] / "build/flappy_env_vector_server"
    with VectorFlappyEnv(executable, num_envs=2) as env:
        results = env.reset_all([99, 100])
        assert all(not result.has_next_pipe for result in results)

        for step in range(80):
            action = 1 if step % 8 == 0 else 0
            results = env.step_batch([action, action])
            for result in results:
                if result.has_next_pipe:
                    assert result.next_pipe_x > 0.0
                    assert result.next_gap_half_height > 0.0
                    print("vector telemetry tests passed")
                    return
        raise AssertionError("no next-pipe telemetry was observed")


if __name__ == "__main__":
    main()
