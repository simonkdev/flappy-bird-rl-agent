import sys
from pathlib import Path

import numpy as np

ENV_PYTHON_DIR = Path(__file__).resolve().parents[1] / "lib/flappy-bird-env/python"
sys.path.insert(0, str(ENV_PYTHON_DIR))

from vector_flappy_env import VectorFlappyEnv


def bird_action(result):
    frame = np.frombuffer(result.observation, dtype=np.uint8).reshape(
        result.height,
        result.width,
    )
    bird_pixels = np.argwhere(frame == 125)
    assert len(bird_pixels) > 0
    return int(bird_pixels[:, 0].mean() > 22)


def run_seeded_episode(env, seed, reset_other_env=False):
    result = env.reset_all([seed] if env.num_envs == 1 else [seed, 99])[0]
    observations = []

    for step in range(100):
        if reset_other_env and step > 0 and step % 9 == 0:
            env.reset_one(1, seed=1000 + step)

        actions = [bird_action(result)]
        if env.num_envs == 2:
            actions.append(0)
        result = env.step_batch(actions)[0]
        observations.append(result.observation)
        if result.terminated:
            break

    return observations


def main():
    executable = Path(__file__).resolve().parents[1] / "lib/flappy-bird-env/build/flappy_env_vector_server"
    with VectorFlappyEnv(executable, num_envs=1) as single_env:
        single_observations = run_seeded_episode(single_env, seed=17)

    with VectorFlappyEnv(executable, num_envs=2) as vector_env:
        vector_observations = run_seeded_episode(
            vector_env,
            seed=17,
            reset_other_env=True,
        )

    assert single_observations == vector_observations
    assert len(single_observations) > 27
    print("vector seed-independence test passed")


if __name__ == "__main__":
    main()
