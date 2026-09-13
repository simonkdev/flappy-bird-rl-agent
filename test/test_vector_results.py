import sys
from pathlib import Path

ENV_PYTHON_DIR = Path(__file__).resolve().parents[1] / "lib/flappy-bird-env/python"
sys.path.insert(0, str(ENV_PYTHON_DIR))

from vector_flappy_env import VectorFlappyEnv


def main() -> None:
    executable = Path(__file__).resolve().parents[1] / "lib/flappy-bird-env/build/flappy_env_vector_server"
    with VectorFlappyEnv(executable, num_envs=2) as env:
        results = env.reset_all([99, 100])
        assert all(not result.passed_pipe and result.score == 0 for result in results)

        results = env.step_batch([1, 1])
        for result in results:
            assert result.shape == (42, 42, 1)
            assert len(result.observation) == 42 * 42
            assert result.simulation_time > 0.0
        print("vector result tests passed")


if __name__ == "__main__":
    main()
