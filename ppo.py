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
from fast_vector_flappy_env import FastVectorFlappyEnv
from vector_flappy_env import VectorFlappyEnv

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

    def __init__(self, debug_window=False, show_game_window=False, env_backend=None):
        self.actor: Actor = Actor(num_actions=config.NUM_ACTIONS)
        self.critic: Critic = Critic()
        self.env_backend = env_backend or config.TRAIN_ENV_BACKEND
        if debug_window or show_game_window or config.NUM_ENVS == 1 or self.env_backend == "subprocess":
            self.vector_env = None
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
        else:
            if self.env_backend == "fast":
                self.vector_env = FastVectorFlappyEnv(
                    num_envs=config.NUM_ENVS,
                    width=config.OBS_WIDTH,
                    height=config.OBS_HEIGHT,
                )
            elif self.env_backend == "cpp_vector":
                self.vector_env = VectorFlappyEnv(
                    num_envs=config.NUM_ENVS,
                    width=config.OBS_WIDTH,
                    height=config.OBS_HEIGHT,
                )
            else:
                raise ValueError(f"Unsupported env_backend: {self.env_backend!r}")
            self.envs = []
            self.env = None
            self.framestack_buffers = [[] for _ in range(config.NUM_ENVS)]
        self.framestack_buffer = self.framestack_buffers[0]
        self.last_trajectory_stats = []

    def training_epoch(self):
        processed_timesteps = self.sample_run()  # python list, 1D with objects in it
        self.actor_training_run(processed_timesteps)
        self.critic_training_run(processed_timesteps)
        return processed_timesteps

    def actor_training_run(self, processed_timesteps):
        states, actions, old_log_probs, advantages, _ = self.timesteps_to_tensors(processed_timesteps)
        indices = np.arange(len(processed_timesteps))

        for _ in range(config.PPO_EPOCHS):
            np.random.shuffle(indices)
            for start in range(0, len(indices), config.PPO_MINIBATCH_SIZE):
                batch_indices = indices[start:start + config.PPO_MINIBATCH_SIZE]
                self.actor_training_step_tensors(
                    tf.gather(states, batch_indices),
                    tf.gather(actions, batch_indices),
                    tf.gather(old_log_probs, batch_indices),
                    tf.gather(advantages, batch_indices),
                )

    def actor_training_step(self,batch):
        with tf.GradientTape() as tape:
            ov_obj = self.overall_objective(batch)
            loss = 0 - ov_obj
        self.actor.update(loss, tape)

    @tf.function(reduce_retracing=True)
    def actor_training_step_tensors(self, states, actions, old_log_probs, advantages):
        with tf.GradientTape() as tape:
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
            loss = -tf.reduce_mean(objective)
        gradients = tape.gradient(loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(gradients, self.actor.trainable_variables))

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
        states, _, _, _, reward_to_go = self.timesteps_to_tensors(processed_timesteps)
        indices = np.arange(len(processed_timesteps))

        for _ in range(config.PPO_EPOCHS):
            np.random.shuffle(indices)
            for start in range(0, len(indices), config.PPO_MINIBATCH_SIZE):
                batch_indices = indices[start:start + config.PPO_MINIBATCH_SIZE]
                self.critic_training_step_tensors(
                    tf.gather(states, batch_indices),
                    tf.gather(reward_to_go, batch_indices),
                )

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

    @tf.function(reduce_retracing=True)
    def critic_training_step_tensors(self, states, reward_to_go):
        with tf.GradientTape() as tape:
            predictions = tf.squeeze(self.critic(states), axis=1)
            loss = tf.reduce_mean(tf.square(reward_to_go - predictions))
        gradients = tape.gradient(loss, self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(gradients, self.critic.trainable_variables))

    def timesteps_to_tensors(self, timesteps):
        states = tf.convert_to_tensor(
            np.stack([timestep.observed_state for timestep in timesteps], axis=0),
            dtype=tf.uint8,
        )
        actions = tf.convert_to_tensor(
            [timestep.action_taken for timestep in timesteps],
            dtype=tf.int32,
        )
        old_log_probs = tf.convert_to_tensor(
            [timestep.sampled_log_probability for timestep in timesteps],
            dtype=tf.float32,
        )
        advantages = tf.convert_to_tensor(
            [timestep.advantage for timestep in timesteps],
            dtype=tf.float32,
        )
        reward_to_go = tf.convert_to_tensor(
            [timestep.reward_to_go for timestep in timesteps],
            dtype=tf.float32,
        )
        return states, actions, old_log_probs, advantages, reward_to_go

    def make_batches(self, timesteps):
        return [
            timesteps[i:i + config.PPO_MINIBATCH_SIZE]
            for i in range(0, len(timesteps), config.PPO_MINIBATCH_SIZE)
        ]

    def close(self):
        if self.vector_env is not None:
            self.vector_env.close()
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
        if self.vector_env is not None:
            return self.collect_vector_trajectories()

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
                env.start_step(actions[i])

            for i, env in enumerate(self.envs):
                if collected_steps >= config.PPO_ROLLOUT_STEPS:
                    break
                result = env.finish_step()
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

    def collect_vector_trajectories(self):
        trajectories = []
        self.last_trajectory_stats = []
        active_trajectories = [[] for _ in range(config.NUM_ENVS)]
        current_results = self.vector_env.reset_all([
            random.randint(0, 120)
            for _ in range(config.NUM_ENVS)
        ])
        episode_steps = [0 for _ in range(config.NUM_ENVS)]
        collected_steps = 0

        while collected_steps < config.PPO_ROLLOUT_STEPS:
            states = [
                self.framestack(result.observation, episode_steps[i], i)
                for i, result in enumerate(current_results)
            ]
            actions, action_log_probs = self.decide_batch(states)
            values = tf.squeeze(self.critic(np.stack(states, axis=0)), axis=1).numpy()
            next_results = self.vector_env.step_batch(actions)

            for i, result in enumerate(next_results):
                if collected_steps >= config.PPO_ROLLOUT_STEPS:
                    break

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
                    current_results[i] = self.vector_env.reset_one(i, seed=random.randint(0, 120))
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

    def decide_deterministic(self, state):
        logits = self.actor(state[np.newaxis, ...])
        action = tf.argmax(logits[0], axis=0)
        return int(action.numpy())

    def decide_batch(self, states):
        logits = self.actor(np.stack(states, axis=0))
        log_probabilities = tf.nn.log_softmax(logits)
        actions = tf.cast(tf.squeeze(tf.random.categorical(logits, 1), axis=1), tf.int32)
        indices = tf.stack([tf.range(tf.shape(actions)[0]), actions], axis=1)
        action_log_probs = tf.gather_nd(log_probabilities, indices)
        return actions.numpy().astype(int).tolist(), action_log_probs.numpy().astype(float).tolist()

    def evaluate_policy(self, episodes=None, max_steps=None, env_backend="cpp_vector", deterministic=True, target_score=None):
        episodes = episodes or config.VALIDATION_EPISODES
        max_steps = max_steps or config.VALIDATION_MAX_STEPS
        target_score = config.VALIDATION_TARGET_SCORE if target_score is None else target_score

        if env_backend == "fast":
            return self._evaluate_vector_policy(
                FastVectorFlappyEnv(
                    num_envs=episodes,
                    width=config.OBS_WIDTH,
                    height=config.OBS_HEIGHT,
                ),
                episodes,
                max_steps,
                deterministic,
                target_score,
            )
        if env_backend == "cpp_vector":
            return self._evaluate_vector_policy(
                VectorFlappyEnv(
                    num_envs=episodes,
                    width=config.OBS_WIDTH,
                    height=config.OBS_HEIGHT,
                ),
                episodes,
                max_steps,
                deterministic,
                target_score,
            )
        if env_backend == "subprocess":
            return self._evaluate_subprocess_policy(episodes, max_steps, deterministic, target_score)
        raise ValueError(f"Unsupported validation env_backend: {env_backend!r}")

    def validate_sim_mismatch(self, episodes=None, max_steps=None, deterministic=True, target_score=None):
        real_stats = self.evaluate_policy(
            episodes=episodes,
            max_steps=max_steps,
            env_backend="cpp_vector",
            deterministic=deterministic,
            target_score=target_score,
        )
        fast_stats = self.evaluate_policy(
            episodes=episodes,
            max_steps=max_steps,
            env_backend="fast",
            deterministic=deterministic,
            target_score=target_score,
        )
        return {
            "real": real_stats,
            "fast": fast_stats,
            "score_gap": fast_stats["avg_score"] - real_stats["avg_score"],
            "step_gap": fast_stats["avg_steps"] - real_stats["avg_steps"],
        }

    def _evaluate_vector_policy(self, env, episodes, max_steps, deterministic, target_score):
        try:
            buffers = [[] for _ in range(episodes)]
            results = env.reset_all([i for i in range(episodes)])
            finished = [False for _ in range(episodes)]
            steps = [0 for _ in range(episodes)]
            scores = [0 for _ in range(episodes)]

            while not all(finished):
                active_indices = [i for i, done in enumerate(finished) if not done]
                states = [
                    self.stack_frame_with_buffer(results[i].observation, steps[i], buffers[i])
                    for i in active_indices
                ]
                actions = self.decide_batch_deterministic(states) if deterministic else self.decide_batch(states)[0]
                batch_actions = [0 for _ in range(episodes)]
                for index, action in zip(active_indices, actions):
                    batch_actions[index] = action

                next_results = env.step_batch(batch_actions)
                for index in active_indices:
                    result = next_results[index]
                    results[index] = result
                    steps[index] += 1
                    scores[index] = result.score
                    if result.terminated or steps[index] >= max_steps or result.score >= target_score:
                        finished[index] = True

            return self._summarize_evaluation(scores, steps, target_score)
        finally:
            env.close()

    def _evaluate_subprocess_policy(self, episodes, max_steps, deterministic, target_score):
        scores = []
        steps = []
        env = FlappyEnv(width=config.OBS_WIDTH, height=config.OBS_HEIGHT)
        try:
            for episode in range(episodes):
                buffer = []
                result = env.reset_result(seed=episode)
                score = 0
                step_count = 0
                for step in range(max_steps):
                    state = self.stack_frame_with_buffer(result.observation, step, buffer)
                    action = self.decide_deterministic(state) if deterministic else self.decide(state)[0]
                    result = env.step_result(action)
                    score = result.score
                    step_count = step + 1
                    if result.terminated or score >= target_score:
                        break
                scores.append(score)
                steps.append(step_count)
            return self._summarize_evaluation(scores, steps, target_score)
        finally:
            env.close()

    def decide_batch_deterministic(self, states):
        logits = self.actor(np.stack(states, axis=0))
        return tf.argmax(logits, axis=1).numpy().astype(int).tolist()

    def stack_frame_with_buffer(self, pixels, step, buffer):
        frame = np.frombuffer(pixels, dtype=np.uint8).reshape(
            config.OBS_HEIGHT,
            config.OBS_WIDTH,
            config.OBS_CHANNELS,
        )
        if step == 0:
            buffer[:] = [frame] * config.FRAME_STACK

        del buffer[0]
        buffer.append(frame)
        return np.concatenate(buffer, axis=-1)

    @staticmethod
    def _summarize_evaluation(scores, steps, target_score):
        return {
            "episodes": len(scores),
            "avg_score": float(np.mean(scores)) if scores else 0.0,
            "max_score": int(max(scores)) if scores else 0,
            "min_score": int(min(scores)) if scores else 0,
            "avg_steps": float(np.mean(steps)) if steps else 0.0,
            "max_steps": int(max(steps)) if steps else 0,
            "target_score": target_score,
            "target_hits": sum(1 for score in scores if score >= target_score),
            "scores": scores,
            "steps": steps,
        }

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
