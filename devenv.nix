{
  pkgs,
  lib,
  config,
  inputs,
  ...
}:

{
  languages.python = {
    enable = true;
    version = "3.13";
    venv.enable = true;
    venv.requirements = ''
      tqdm
    '';
  };

  packages = with pkgs; [
    cmake
    wayland
    wayland-scanner
    libxkbcommon
    libffi
    glfw
    waylandpp
    gcc
    python313Packages.tensorflow
    python313Packages.numpy
    python313Packages.keras
    python313Packages.tkinter
  ];

  languages.cplusplus.enable = true;

  env.LD_LIBRARY_PATH = "${pkgs.wayland}/lib:${pkgs.libxkbcommon}/lib";

  tasks = {
    "flappyrl:libdemo".exec =
      "python3 lib/flappy-bird-env/demo.py --show-game-window --debug-window --infinite --real-time";

    "flappyrl:validate-training".exec = ''
      cmake -S lib/flappy-bird-env -B lib/flappy-bird-env/build
      cmake --build lib/flappy-bird-env/build --target flappy_env_server flappy_env_vector_server
      python3 training.py \
        --epochs 100 \
        --env-backend cpp_vector \
        --rollout-steps 8192 \
        --num-envs 32 \
        --ppo-epochs 4 \
        --minibatch-size 512 \
        --validate-every 10 \
        --validation-episodes 20 \
        --validation-max-steps 10000 \
        --validation-target-score 500
    '';
  };

}
