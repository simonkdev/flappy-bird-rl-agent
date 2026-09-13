import numpy as np
import tensorflow as tf

from src import config
from src.ppo import PPO, processed_timestep


def make_timesteps(count):
    state = np.zeros(
        (
            config.OBS_HEIGHT,
            config.OBS_WIDTH,
            config.OBS_CHANNELS * config.FRAME_STACK,
        ),
        dtype=np.uint8,
    )
    return [
        processed_timestep(
            sampled_log_probability=0.0,
            advantage=1.0,
            observed_state=state,
            action_taken=0,
            reward_to_go=0.0,
        )
        for _ in range(count)
    ]


def main():
    original_epochs = config.PPO_EPOCHS
    original_minibatch_size = config.PPO_MINIBATCH_SIZE
    original_target_kl = config.PPO_TARGET_KL
    config.PPO_EPOCHS = 3
    config.PPO_MINIBATCH_SIZE = 2
    config.PPO_TARGET_KL = 0.01
    ppo = PPO(env_backend="fast")
    calls = []

    try:
        ppo.shuffled_indices = lambda size: np.arange(size)

        def training_step(*_):
            calls.append(None)
            kl = 0.001 if len(calls) == 1 else 0.02
            return tuple(tf.constant(value, dtype=tf.float32) for value in (0.0, 0.0, kl, 0.0))

        ppo.actor_training_step_tensors = training_step
        metrics = ppo.actor_training_run(make_timesteps(6))
        assert len(calls) == 2
        assert metrics["early_stop"]
        assert metrics["ppo_passes"] == 0
        assert metrics["approx_kl"] > config.PPO_TARGET_KL
    finally:
        ppo.close()
        config.PPO_EPOCHS = original_epochs
        config.PPO_MINIBATCH_SIZE = original_minibatch_size
        config.PPO_TARGET_KL = original_target_kl

    print("actor minibatch KL-stop tests passed")


if __name__ == "__main__":
    main()
