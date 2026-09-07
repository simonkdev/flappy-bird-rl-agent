# Installation instructions

On Linux/MacOS:

1. Install CMake: https://cmake.org/download/
2. Clone the repository:
   ```
   git clone --recurse-submodules https://github.com/TS-Code-Studios/flappy-bird && cd flappy-bird
   ```
3. Create build files:
   ```
   cmake -B build && cd build
   ```
4. Build the project (using Unix Makefile):
   ```
   make
   ```
5. Run the executable:
   ```
   ./flappy_bird
   ```

## Python environment interface

This repository also includes a Python-controllable environment server for reinforcement-learning experiments. See [docs/rl_environment.md](docs/rl_environment.md) for build, demo, API, observation, action, timing, and debug-window details.
