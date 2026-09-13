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

    "flappyrl:train-fixed-static-gamma".exec = ''
      cmake -S lib/flappy-bird-env -B lib/flappy-bird-env/build
      cmake --build lib/flappy-bird-env/build --target flappy_env_server flappy_env_vector_server
      python3 -u -m train.training \
        --epochs 310 \
        --env-backend cpp_vector \
        --rollout-steps 16384 \
        --num-envs 32 \
        --ppo-epochs 4 \
        --critic-ppo-epochs 8 \
        --minibatch-size 256 \
        --learning-rate 0.00005 \
        --critic-learning-rate 0.0005 \
        --target-kl 0.01 \
        --gamma 0.995 \
        --gae-lambda 0.95 \
        --entropy-start 0.01 \
        --entropy-end 0.003 \
        --entropy-decay-epochs 100 \
        --seed 20260912 \
        --train-seed-min 0 \
        --train-seed-max 2000000000 \
        --validate-every 10 \
        --validation-episodes 20 \
        --validation-max-steps 10000 \
        --validation-target-score 500 \
        --validation-seed-start 3000000000 \
        --checkpoint-dir checkpoints/spatial-cnn-fixed-observation-static-gamma-995-low-actor-lr-entropy-warmup \
        --checkpoint-every 5 \
        --checkpoint-keep 10 \
        --resume-from checkpoints/spatial-cnn-fixed-observation-static-gamma-995-low-actor-lr-entropy-warmup/latest \
        --min-available-memory-mib 2048 \
        --oom-score-adj 500
    '';

    "flappyrl:train-fixed-observation-stable-long-horizon".exec = ''
      cmake -S lib/flappy-bird-env -B lib/flappy-bird-env/build
      cmake --build lib/flappy-bird-env/build --target flappy_env_server flappy_env_vector_server
      python3 -u -m train.training \
        --epochs 300 \
        --env-backend cpp_vector \
        --rollout-steps 16384 \
        --num-envs 32 \
        --ppo-epochs 4 \
        --critic-ppo-epochs 8 \
        --minibatch-size 512 \
        --learning-rate 0.000025 \
        --critic-learning-rate 0.00025 \
        --target-kl 0.006 \
        --gamma 0.997 \
        --gae-lambda 0.97 \
        --entropy-start 0.004 \
        --entropy-end 0.004 \
        --entropy-decay-epochs 1 \
        --seed 20260912 \
        --train-seed-min 0 \
        --train-seed-max 2000000000 \
        --validate-every 10 \
        --validation-episodes 20 \
        --validation-max-steps 10000 \
        --validation-target-score 500 \
        --validation-seed-start 3000000000 \
        --checkpoint-dir checkpoints/spatial-cnn-fixed-observation-stable-long-horizon-from-290 \
        --checkpoint-every 5 \
        --checkpoint-keep 10 \
        --resume-from checkpoints/spatial-cnn-fixed-observation-static-gamma-995-low-actor-lr-entropy-warmup/best/ckpt-290 \
        --min-available-memory-mib 2048 \
        --oom-score-adj 500
    '';
  };

}
