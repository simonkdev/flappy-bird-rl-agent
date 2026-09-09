"""Environment adapters provided by the bundled Flappy Bird library."""

import sys
from pathlib import Path


ENVIRONMENT_PYTHON_DIR = Path(__file__).resolve().parents[1] / "lib/flappy-bird-env/python"
if str(ENVIRONMENT_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(ENVIRONMENT_PYTHON_DIR))

from fast_vector_flappy_env import FastVectorFlappyEnv
from flappy_env import FlappyEnv
from vector_flappy_env import VectorFlappyEnv
