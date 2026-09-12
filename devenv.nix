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
      run_segment() {
        local resume_args=()
        if [ -n "$1" ]; then
          resume_args=(--resume-from "$1")
        fi
        python3 -u -m train.training \
          --epochs 20 \
          --env-backend cpp_vector \
          --rollout-steps 16384 \
          --num-envs 32 \
          --ppo-epochs 4 \
          --critic-ppo-epochs 8 \
          --minibatch-size 512 \
          --learning-rate 0.00005 \
          --critic-learning-rate 0.0005 \
          --target-kl 0.02 \
          --gamma 0.9999 \
          --gae-lambda 0.95 \
          --entropy-start 0.003 \
          --entropy-end 0.003 \
          --entropy-decay-epochs 1 \
          --seed 20260912 \
          --train-seed-min 0 \
          --train-seed-max 2000000000 \
          --validate-every 10 \
          --validation-episodes 20 \
          --validation-max-steps 10000 \
          --validation-target-score 500 \
          --validation-seed-start 3000000000 \
          --checkpoint-dir checkpoints/spatial-cnn-low-actor-lr-gamma-9999-extension-from-200 \
          --checkpoint-every 10 \
          --checkpoint-keep 10 \
          "''${resume_args[@]}"
      }
      run_segment checkpoints/spatial-cnn-low-actor-lr-gamma-curriculum-from-scratch/latest/ckpt-200
      run_segment checkpoints/spatial-cnn-low-actor-lr-gamma-9999-extension-from-200/latest/ckpt-220
      run_segment checkpoints/spatial-cnn-low-actor-lr-gamma-9999-extension-from-200/latest/ckpt-240
      run_segment checkpoints/spatial-cnn-low-actor-lr-gamma-9999-extension-from-200/latest/ckpt-260
      run_segment checkpoints/spatial-cnn-low-actor-lr-gamma-9999-extension-from-200/latest/ckpt-280
      run_segment checkpoints/spatial-cnn-low-actor-lr-gamma-9999-extension-from-200/latest/ckpt-300
      run_segment checkpoints/spatial-cnn-low-actor-lr-gamma-9999-extension-from-200/latest/ckpt-320
      run_segment checkpoints/spatial-cnn-low-actor-lr-gamma-9999-extension-from-200/latest/ckpt-340
      run_segment checkpoints/spatial-cnn-low-actor-lr-gamma-9999-extension-from-200/latest/ckpt-360
      run_segment checkpoints/spatial-cnn-low-actor-lr-gamma-9999-extension-from-200/latest/ckpt-380
    '';
  };

}
