from critic import Critic
from actor import Actor
from dataclasses import dataclass
import numpy as np
import random
import config
import sys
import tensorflow as tf
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "lib/flappy-bird-env/python"))
from flappy_env import FlappyEnv

@dataclass
class processed_timestep:
    sampled_log_probability: float
    advantage: float
    observed_state: np.ndarray
    action_taken: int
    reward_to_go: float
    sampled_critic_value: float

@dataclass
class timestep:
    observed_state: np.ndarray
    action_taken: int
    sampled_log_probability: float
    reward: float
    sampled_critic_value: float
    reward_to_go: float

class PPO:
#    name: str = ""

    def __init__(self, debug_window=False, show_game_window=False):
        self.actor: Actor = Actor(num_actions=config.NUM_ACTIONS)
        self.critic: Critic = Critic()
        self.envs = [
            FlappyEnv(
                width=config.OBS_WIDTH,
                height=config.OBS_HEIGHT,
                debug_window=debug_window,
                show_game_window=show_game_window,
            )
            for _ in range(config.NUM_ENVS)
        ]
        self.env = self.envs[0]
        self.framestack_buffers = [[] for _ in self.envs]
        self.framestack_buffer = self.framestack_buffers[0]
        self.last_trajectory_stats = []

    def training_epoch(self):
        processed_timesteps = self.sample_run()  # python list, 1D with objects in it
        self.actor_training_run(processed_timesteps)
        self.critic_training_run(processed_timesteps)
        return processed_timesteps

    def actor_training_run(self, processed_timesteps):
        timesteps = processed_timesteps
        random.shuffle(timesteps)

        batches = self.make_batches(timesteps)

        for i in range(config.PPO_EPOCHS):
            for batch in batches:
                self.actor_training_step(batch)

    def actor_training_step(self,batch):
        with tf.GradientTape() as tape:
            ov_obj = self.overall_objective(batch)
            loss = 0 - ov_obj
        self.actor.update(loss, tape)

    def overall_objective(self, batch):
        states = np.stack([timestep.observed_state for timestep in batch], axis=0)
        actions = tf.convert_to_tensor(
            [timestep.action_taken for timestep in batch],
            dtype=tf.int32,
        )
        old_log_probs = tf.convert_to_tensor(
            [timestep.sampled_log_probability for timestep in batch],
            dtype=tf.float32,
        )
        advantages = tf.convert_to_tensor(
            [timestep.advantage for timestep in batch],
            dtype=tf.float32,
        )

        logits = self.actor(states)
        log_probabilities = tf.nn.log_softmax(logits)
        indices = tf.stack([tf.range(tf.shape(actions)[0]), actions], axis=1)
        new_log_probs = tf.gather_nd(log_probabilities, indices)
        ratio = tf.exp(new_log_probs - old_log_probs)
        clipped_ratio = tf.clip_by_value(
            ratio,
            1.0 - config.PPO_CLIP_EPSILON,
            1.0 + config.PPO_CLIP_EPSILON,
        )
        objective = tf.minimum(
            ratio * advantages,
            clipped_ratio * advantages,
        )
        return tf.reduce_mean(objective)

    def critic_training_run(self, processed_timesteps):
        timesteps = processed_timesteps
        random.shuffle(timesteps)

        batches = self.make_batches(timesteps)

        for i in range(config.PPO_EPOCHS):
            for batch in batches:
                self.critic_training_step(batch)

    def critic_training_step(self, batch):
        with tf.GradientTape() as tape:
            states = np.stack([timestep.observed_state for timestep in batch], axis=0)
            rtgs = tf.convert_to_tensor(
                [timestep.reward_to_go for timestep in batch],
                dtype=tf.float32,
            )
            predictions = tf.squeeze(self.critic(states), axis=1)
            loss = tf.reduce_mean(tf.square(rtgs - predictions))
        self.critic.update(loss, tape)

    def make_batches(self, timesteps):
        return [
            timesteps[i:i + config.PPO_MINIBATCH_SIZE]
            for i in range(0, len(timesteps), config.PPO_MINIBATCH_SIZE)
        ]

    def close(self):
        for env in self.envs:
            env.close()

    def sample_run(self):
        trajectories = self.collect_trajectories()
        timesteps = self.process_trajectories(trajectories)
        return timesteps

    def process_trajectories(self, trajectories):
        # input is a 2D-array of timestep objects
        processed_timesteps = []

        for trajectory in trajectories:
            reward_buffer = 0.0

            for timestep in reversed(trajectory):
                reward_to_go = timestep.reward + (config.GAMMA * reward_buffer)
                reward_buffer = reward_to_go

                processed_timesteps.append(
                    processed_timestep(
                        sampled_log_probability=timestep.sampled_log_probability,
                        advantage=reward_to_go - timestep.sampled_critic_value,
                        observed_state=timestep.observed_state,
                        action_taken=timestep.action_taken,
                        reward_to_go=reward_to_go,
                        sampled_critic_value=timestep.sampled_critic_value,
                    )
                )

        return self.normalize_advantages(processed_timesteps)

    def normalize_advantages(self, processed_timesteps):
        if not processed_timesteps:
            return processed_timesteps

        advantages = np.array(
            [timestep.advantage for timestep in processed_timesteps],
            dtype=np.float32,
        )
        mean = np.mean(advantages)
        std = np.std(advantages)
        if std < 1e-8:
            std = 1.0

        for timestep in processed_timesteps:
            timestep.advantage = (timestep.advantage - mean) / std

        return processed_timesteps

    def collect_trajectories(self):
        trajectories = []
        self.last_trajectory_stats = []
        active_trajectories = [[] for _ in self.envs]
        current_results = [
            env.reset_result(seed=random.randint(0, 120))
            for env in self.envs
        ]
        episode_steps = [0 for _ in self.envs]
        collected_steps = 0

        while collected_steps < config.PPO_ROLLOUT_STEPS:
            states = [
                self.framestack(result.observation, episode_steps[i], i)
                for i, result in enumerate(current_results)
            ]
            actions, action_log_probs = self.decide_batch(states)
            values = tf.squeeze(self.critic(np.stack(states, axis=0)), axis=1).numpy()

            for i, env in enumerate(self.envs):
                if collected_steps >= config.PPO_ROLLOUT_STEPS:
                    break

                result = env.step_result(actions[i])
                if result.terminated:
                    reward = config.REWARD_DIE
                elif result.passed_pipe:
                    reward = config.REWARD_PASSED_PIPE
                else:
                    reward = config.REWARD_STD

                active_trajectories[i].append(
                    timestep(
                        observed_state=states[i],
                        action_taken=actions[i],
                        sampled_log_probability=action_log_probs[i],
                        reward=reward,
                        sampled_critic_value=float(values[i]),
                        reward_to_go=0.0
                    )
                )
                collected_steps += 1

                episode_steps[i] += 1
                current_results[i] = result
                if result.terminated or episode_steps[i] >= config.MAX_NUM_STEPS:
                    trajectories.append(active_trajectories[i])
                    self.record_trajectory_stats(active_trajectories[i], result.terminated)
                    active_trajectories[i] = []
                    current_results[i] = env.reset_result(seed=random.randint(0, 120))
                    episode_steps[i] = 0

        for trajectory in active_trajectories:
            if trajectory:
                trajectories.append(trajectory)
                self.record_trajectory_stats(trajectory, False)

        return trajectories

    def record_trajectory_stats(self, trajectory, terminated):
        self.last_trajectory_stats.append({
            "length": len(trajectory),
            "reward": sum(timestep.reward for timestep in trajectory),
            "pipes": sum(1 for timestep in trajectory if timestep.reward == config.REWARD_PASSED_PIPE),
            "terminated": terminated,
        })

    def collect_trajectory(self):
        seed = random.randint(0, 120)
        result = self.env.reset_result(seed=seed)
        timesteps = []

        for step in range(config.MAX_NUM_STEPS):
            pixels = result.observation
            state = self.framestack(pixels, step)
            action, action_log_prob = self.decide(state)
            critic_value = float(self.critic(state[np.newaxis, ...])[0, 0].numpy())
            result = self.env.step_result(action)
            passed = result.passed_pipe
            terminated = result.terminated
            if terminated:
                reward = config.REWARD_DIE
            elif passed:
                reward = config.REWARD_PASSED_PIPE
            else:
                reward = config.REWARD_STD
            current_timestep = timestep(
                observed_state=state,
                action_taken=action,
                sampled_log_probability=action_log_prob,
                reward=reward,
                sampled_critic_value=critic_value,
                reward_to_go=0.0
            )
            timesteps.append(current_timestep)
            if terminated:
                return timesteps
        return timesteps

    def framestack(self, pixels, step, env_index=0):
        frame = np.frombuffer(pixels, dtype=np.uint8).reshape(
            config.OBS_HEIGHT,
            config.OBS_WIDTH,
            config.OBS_CHANNELS,
        )
        buffer = self.framestack_buffers[env_index]
        if step == 0:
            buffer[:] = [frame] * config.FRAME_STACK

        del buffer[0]
        buffer.append(frame)
        return np.concatenate(buffer, axis=-1)

    def decide(self, state):
        logits = self.actor(state[np.newaxis, ...])
        log_probabilities = tf.nn.log_softmax(logits)[0]
        probabilities = tf.exp(log_probabilities).numpy()
        action = np.random.choice(config.NUM_ACTIONS, p=probabilities)
        return int(action), float(log_probabilities[action].numpy())

    def decide_batch(self, states):
        logits = self.actor(np.stack(states, axis=0))
        log_probabilities = tf.nn.log_softmax(logits).numpy()
        probabilities = np.exp(log_probabilities)
        actions = [
            int(np.random.choice(config.NUM_ACTIONS, p=probabilities[i]))
            for i in range(len(states))
        ]
        action_log_probs = [
            float(log_probabilities[i, action])
            for i, action in enumerate(actions)
        ]
        return actions, action_log_probs

    def testdump(self):
        observation = np.random.rand(1, 42, 42, 5)*255
        print(self.actor(observation))
        print(self.critic(observation))

if __name__ == "__main__":
    ppotest = PPO()
    try:
        ppotest.testdump()
    finally:
        ppotest.close()
