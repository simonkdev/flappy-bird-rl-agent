import sys
from pathlib import Path


environment_python_path = Path(__file__).resolve().parents[1] / "lib/flappy-bird-env/python"
if str(environment_python_path) not in sys.path:
    sys.path.insert(0, str(environment_python_path))

from fast_vector_flappy_env import FastVectorFlappyEnv
from flappy_env import FlappyEnv
from vector_flappy_env import VectorFlappyEnv
