# Flappy Bird Python Environment Interface

This project exposes the existing C++ Flappy Bird game as a small Python-controllable environment. The normal `flappy_bird` executable remains the manual playable game. The RL interface is a separate executable, `flappy_env_server`, controlled by a Python subprocess wrapper.

## Build

From the RL project root:

```sh
cmake -B lib/flappy-bird-env/build -S lib/flappy-bird-env
cmake --build lib/flappy-bird-env/build
```

The environment server will be created at `lib/flappy-bird-env/build/flappy_env_server`.

## Python Demo

From the RL project root:

```sh
python3 lib/flappy-bird-env/demo.py
```

Run a specific number of scripted episodes:

```sh
python3 lib/flappy-bird-env/demo.py --episodes 10
```

Run scripted episodes until interrupted:

```sh
python3 lib/flappy-bird-env/demo.py --infinite
```

Pace the demo to simulated real time, useful when showing windows:

```sh
python3 lib/flappy-bird-env/demo.py --show-game-window --debug-window --real-time --infinite
```

Optional visual debugging:

```sh
python3 lib/flappy-bird-env/demo.py --debug-window --show-game-window
```

`--debug-window` shows the exact low-resolution grayscale observation that Python receives, scaled up. `--show-game-window` also shows and presents the normal game window from the same simulation state during each `reset()`/`step()`. Training mode should omit both flags.

## Python API

```python
import sys
sys.path.insert(0, "lib/flappy-bird-env/python")

from flappy_env import FlappyEnv

with FlappyEnv(seed=123) as env:
    obs = env.reset()

    done = False
    while not done:
        action = 0
        obs, reward, done, info = env.step(action)
```

Actions:

- `0`: do not flap
- `1`: flap

Invalid actions raise `ValueError` in Python and return `ERR` from the C++ server.

## Observation

Each observation is one grayscale frame from the current game state. The RL pass intentionally does not sample the game's textures; it renders a high-contrast silhouette view so decorative texture detail does not leak into the agent input.

- Shape: `(height, width, 1)` in Python metadata
- Default size: `42 x 42 x 1`
- Storage: Python `bytes`
- Type: raw unsigned 8-bit pixels
- Range: `0..255`
- Layout: row-major, top row first
- Background/art/text: black/omitted
- Pipes: white, value `255`
- Bird: solid gray, value `125`

The C++ side returns individual frames. Keep five-frame history in Python by storing the last five returned observations.

## Step Timing

`step(action)` advances a fixed amount of simulated time. Defaults:

- `fixedDeltaTime = 1 / 60`
- `ticksPerStep = 4`

The Python process may take any amount of wall-clock time between calls. Simulation advances only when `step()` is called.

## Reward and Info

The built-in reward is intentionally minimal and game-centric:

- `+1` for each newly passed pipe
- `-1` when the bird dies
- `0` otherwise

The Python `info` dict includes:

- `alive`
- `score`
- `passed_pipe`
- `simulation_time`
- `shape`
- `dtype`

Use these fields to define your own RL reward/state handling outside the C++ game.

## Headless and Debug Modes

The server defaults to no visible game/debug windows, suitable for automated sampling. It still creates a hidden OpenGL context, so the machine needs a working OpenGL/GLFW display backend.

Debug flags:

- `--debug-window`: display the low-resolution RL observation.
- `--show-game-window`: display and swap the normal game window during environment stepping.

These modes use one simulated game state. The debug observation is not a second game.

## Multiple Environments

The Python wrapper starts one `flappy_env_server` subprocess per `FlappyEnv`. Multiple independent environments can be created by constructing multiple wrappers. Each subprocess owns its own game world, bird, pipes, score, timing, and random seed.

## C++ Protocol

The server reads one command per line:

- `RESET [seed]`
- `STEP action`
- `INFO`
- `CLOSE`

`RESET` and `STEP` return one `OK` metadata line ending with the raw frame byte count, followed immediately by that many raw grayscale bytes on stdout. This avoids base64 overhead while keeping the command protocol simple.
