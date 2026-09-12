import tempfile

import tensorflow as tf

from src.ppo import PPO
from train.training import create_checkpoint_managers


def main():
    with tempfile.TemporaryDirectory() as checkpoint_dir:
        original = PPO(env_backend="fast")
        try:
            checkpoint, managers = create_checkpoint_managers(original, checkpoint_dir, 1)
            original.sample_training_seed()
            original.next_stateless_seed(original.action_rng)
            original.shuffled_indices(16)
            checkpoint.epoch.assign(1)
            checkpoint_path = managers["latest"].save(checkpoint_number=1)

            expected_seed = original.sample_training_seed()
            expected_action_seed = original.next_stateless_seed(original.action_rng).numpy()
            expected_indices = original.shuffled_indices(16)

            restored = PPO(env_backend="fast")
            try:
                restored_checkpoint, _ = create_checkpoint_managers(restored, checkpoint_dir, 1)
                status = restored_checkpoint.restore(checkpoint_path)
                status.assert_existing_objects_matched()

                assert restored.sample_training_seed() == expected_seed
                assert (restored.next_stateless_seed(restored.action_rng).numpy() == expected_action_seed).all()
                assert (restored.shuffled_indices(16) == expected_indices).all()
            finally:
                restored.close()
        finally:
            original.close()

    print("checkpoint RNG tests passed")


if __name__ == "__main__":
    main()
