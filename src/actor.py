import tensorflow as tf
from tensorflow.keras import layers

from . import config


class Actor(tf.keras.Model):

    def __init__(self, num_actions):
        super().__init__()

        self.cnn = tf.keras.Sequential([
            layers.Conv2D(
                config.CNN_FILTERS[0],
                config.CNN_KERNEL_SIZES[0],
                strides=config.CNN_STRIDES[0],
                activation="relu"
            ),
            layers.Conv2D(
                config.CNN_FILTERS[1],
                config.CNN_KERNEL_SIZES[1],
                strides=config.CNN_STRIDES[1],
                activation="relu"
            ),
            layers.Conv2D(
                config.CNN_FILTERS[2],
                config.CNN_KERNEL_SIZES[2],
                strides=config.CNN_STRIDES[2],
                activation="relu"
            ),
            layers.Flatten(),
        ])

        self.fc = layers.Dense(
            config.FC_UNITS,
            activation="relu"
        )

        self.policy = layers.Dense(num_actions)

        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=config.LEARNING_RATE
        )

    def call(self, observation, return_probs=False):
        x = tf.cast(observation, tf.float32) / config.PIXEL_SCALE

        x = self.cnn(x)
        x = self.fc(x)

        logits = self.policy(x)

        if return_probs:
            return tf.nn.softmax(logits)
        return logits
