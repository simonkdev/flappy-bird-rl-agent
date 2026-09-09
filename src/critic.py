import tensorflow as tf
from tensorflow.keras import layers

from . import config


class Critic(tf.keras.Model):

    def __init__(self):
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

        # The critic outputs one value: V(s)
        self.value = layers.Dense(1)

        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=config.CRITIC_LEARNING_RATE
        )

    def call(self, observation):
        x = tf.cast(observation, tf.float32) / config.PIXEL_SCALE

        x = self.cnn(x)
        x = self.fc(x)

        value = self.value(x)

        return value
