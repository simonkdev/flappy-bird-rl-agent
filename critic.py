import tensorflow as tf
from tensorflow.keras import layers

import config


class Critic(tf.keras.Model):

    def __init__(self):
        super().__init__()

        self.mlp = tf.keras.Sequential([
            layers.Flatten(),

            layers.Dense(
                config.CRITIC_HIDDEN_UNITS[0],
                activation="relu"
            ),

            layers.Dense(
                config.CRITIC_HIDDEN_UNITS[1],
                activation="relu"
            ),

            layers.Dense(
                config.CRITIC_HIDDEN_UNITS[2],
                activation="relu"
            ),
        ])

        # The critic outputs one value: V(s)
        self.value = layers.Dense(1)

        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=config.CRITIC_LEARNING_RATE
        )

    def call(self, observation):
        x = tf.cast(observation, tf.float32)

        x = self.mlp(x)

        value = self.value(x)

        return value

    def forward_prop(self, observation):
        return self.call(observation)

    def update(self, loss, tape):
        gradients = tape.gradient(
            loss,
            self.trainable_variables
        )

        self.optimizer.apply_gradients(
            zip(gradients, self.trainable_variables)
        )
