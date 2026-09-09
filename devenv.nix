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
      python3 -u training.py \
        --epochs 500 \
        --env-backend cpp_vector \
        --rollout-steps 16384 \
        --num-envs 32 \
        --ppo-epochs 4 \
        --critic-ppo-epochs 6 \
        --minibatch-size 512 \
        --learning-rate 0.0002 \
        --critic-learning-rate 0.0002 \
        --target-kl 0.020 \
        --entropy-start 0.01 \
        --entropy-end 0.003 \
        --entropy-decay-epochs 300 \
        --validate-every 10 \
        --validation-episodes 20 \
        --validation-max-steps 10000 \
        --validation-target-score 500 \
        --checkpoint-dir checkpoints/original-reward-stability-from-160 \
        --checkpoint-every 25 \
        --resume-from checkpoints/validate-training/best/ckpt-160
    '';
  };

}
