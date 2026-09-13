from dataclasses import dataclass

import numpy as np
import tensorflow as tf

from . import config
from .actor import Actor
from .critic import Critic
from .environment import FastVectorFlappyEnv, FlappyEnv, VectorFlappyEnv

@dataclass
class processed_timestep:
    sampled_log_probability: float
    advantage: float
    observed_state: np.ndarray
    action_taken: int
    reward_to_go: float

@dataclass
class timestep:
    observed_state: np.ndarray
    action_taken: int
    sampled_log_probability: float
    reward: float
    sampled_critic_value: float
    passed_pipe: bool = False
    terminated: bool = False
    next_critic_value: float = 0.0

class PPO:

    def __init__(
        self,
        debug_window=False,
        show_game_window=False,
        env_backend=None,
        num_envs=None,
    ):
        self.actor: Actor = Actor(num_actions=config.NUM_ACTIONS)
        self.critic: Critic = Critic()
        self.env_backend = env_backend or config.TRAIN_ENV_BACKEND
        self.num_envs = config.NUM_ENVS if num_envs is None else num_envs
        if self.num_envs < 1:
            raise ValueError("num_envs must be positive")
        if (
            debug_window
            or show_game_window
            or self.num_envs == 1
            or self.env_backend == "subprocess"
        ):
            self.vector_env = None
            self.envs = [
                FlappyEnv(
                    width=config.OBS_WIDTH,
                    height=config.OBS_HEIGHT,
                    debug_window=debug_window,
                    show_game_window=show_game_window,
                )
                for _ in range(self.num_envs)
            ]
            self.framestack_buffers = [[] for _ in self.envs]
        else:
            if self.env_backend == "fast":
                self.vector_env = FastVectorFlappyEnv(
                    num_envs=self.num_envs,
                    width=config.OBS_WIDTH,
                    height=config.OBS_HEIGHT,
                )
            elif self.env_backend == "cpp_vector":
                self.vector_env = VectorFlappyEnv(
                    num_envs=self.num_envs,
                    width=config.OBS_WIDTH,
                    height=config.OBS_HEIGHT,
                )
            else:
                raise ValueError(f"Unsupported env_backend: {self.env_backend!r}")
            self.envs = []
            self.framestack_buffers = [[] for _ in range(self.num_envs)]
        self.last_trajectory_stats = []
        self.last_training_metrics = {}
        self.current_results = None
        self.episode_steps = None
        self.training_seed_rng = tf.random.Generator.from_seed(config.TRAINING_SEED)
        self.action_rng = tf.random.Generator.from_seed(config.TRAINING_SEED + 1)
        self.shuffle_rng = tf.random.Generator.from_seed(config.TRAINING_SEED + 2)
        self.entropy_coefficient = tf.Variable(
            config.PPO_ENTROPY_COEFFICIENT_START,
            dtype=tf.float32,
            trainable=False,
        )

    def set_entropy_coefficient(self, value):
        self.entropy_coefficient.assign(float(value))

    def training_epoch(self):
        processed_timesteps = self.sample_run()
        actor_metrics = self.actor_training_run(processed_timesteps)
        critic_metrics = self.critic_training_run(processed_timesteps)
        self.last_training_metrics = actor_metrics | critic_metrics
        return processed_timesteps

    def actor_training_run(self, processed_timesteps):
        states, actions, old_log_probs, advantages, _ = self.timesteps_to_tensors(
            processed_timesteps
        )
        indices = np.arange(len(processed_timesteps))
        metrics = []
        early_stopped = False
        completed_passes = 0
        max_batch_kl = 0.0
        stop_batch_kl = None

        for _ in range(config.PPO_EPOCHS):
            indices = self.shuffled_indices(len(processed_timesteps))
            for start in range(0, len(indices), config.PPO_MINIBATCH_SIZE):
                batch_indices = indices[start:start + config.PPO_MINIBATCH_SIZE]
                batch_metrics = self.actor_training_step_tensors(
                    tf.gather(states, batch_indices),
                    tf.gather(actions, batch_indices),
                    tf.gather(old_log_probs, batch_indices),
                    tf.gather(advantages, batch_indices),
                )
                metrics.append([float(metric.numpy()) for metric in batch_metrics])
                batch_kl = float(batch_metrics[2].numpy())
                max_batch_kl = max(max_batch_kl, batch_kl)
                if batch_kl >= config.PPO_TARGET_KL:
                    early_stopped = True
                    stop_batch_kl = batch_kl
                    break
            if early_stopped:
                break
            completed_passes += 1

        policy_loss, entropy, approx_kl, clip_fraction = np.mean(metrics, axis=0)
        return {
            "policy_loss": float(policy_loss),
            "actor_entropy": float(entropy),
            "approx_kl": float(approx_kl),
            "clip_fraction": float(clip_fraction),
            "ppo_passes": completed_passes,
            "early_stop": early_stopped,
            "max_batch_kl": max_batch_kl,
            "stop_batch_kl": stop_batch_kl,
        }

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
            probabilities = tf.exp(log_probabilities)
            entropy = -tf.reduce_sum(probabilities * log_probabilities, axis=1)
            mean_entropy = tf.reduce_mean(entropy)
            policy_loss = -tf.reduce_mean(objective)
            loss = policy_loss - (self.entropy_coefficient * mean_entropy)
        gradients = tape.gradient(loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(gradients, self.actor.trainable_variables))
        updated_logits = self.actor(states)
        updated_log_probabilities = tf.nn.log_softmax(updated_logits)
        updated_log_probs = tf.gather_nd(updated_log_probabilities, indices)
        updated_ratio = tf.exp(updated_log_probs - old_log_probs)
        updated_log_ratio = updated_log_probs - old_log_probs
        approx_kl = tf.reduce_mean((updated_ratio - 1.0) - updated_log_ratio)
        clip_fraction = tf.reduce_mean(
            tf.cast(tf.abs(updated_ratio - 1.0) > config.PPO_CLIP_EPSILON, tf.float32)
        )
        return policy_loss, mean_entropy, approx_kl, clip_fraction

    def critic_training_run(self, processed_timesteps):
        states, _, _, _, reward_to_go = self.timesteps_to_tensors(processed_timesteps)
        indices = np.arange(len(processed_timesteps))
        losses = []

        for _ in range(config.CRITIC_PPO_EPOCHS):
            indices = self.shuffled_indices(len(processed_timesteps))
            for start in range(0, len(indices), config.PPO_MINIBATCH_SIZE):
                batch_indices = indices[start:start + config.PPO_MINIBATCH_SIZE]
                loss = self.critic_training_step_tensors(
                    tf.gather(states, batch_indices),
                    tf.gather(reward_to_go, batch_indices),
                )
                losses.append(float(loss.numpy()))

        predictions = []
        for start in range(0, len(processed_timesteps), config.PPO_MINIBATCH_SIZE):
            batch_states = states[start:start + config.PPO_MINIBATCH_SIZE]
            predictions.extend(tf.squeeze(self.critic(batch_states), axis=1).numpy())
        targets = reward_to_go.numpy()
        target_variance = np.var(targets)
        if target_variance < 1e-8:
            explained_variance = 0.0
        else:
            explained_variance = 1.0 - np.var(targets - predictions) / target_variance
        return {
            "value_loss": float(np.mean(losses)),
            "explained_variance": float(explained_variance),
        }

    @tf.function(reduce_retracing=True)
    def critic_training_step_tensors(self, states, reward_to_go):
        with tf.GradientTape() as tape:
            predictions = tf.squeeze(self.critic(states), axis=1)
            loss = tf.reduce_mean(tf.square(reward_to_go - predictions))
        gradients = tape.gradient(loss, self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(gradients, self.critic.trainable_variables))
        return loss

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
            gae_buffer = 0.0

            for timestep in reversed(trajectory):
                nonterminal = 0.0 if timestep.terminated else 1.0
                delta = (
                    timestep.reward
                    + config.GAMMA * timestep.next_critic_value * nonterminal
                    - timestep.sampled_critic_value
                )
                advantage = delta + config.GAMMA * config.GAE_LAMBDA * nonterminal * gae_buffer
                gae_buffer = advantage
                value_target = advantage + timestep.sampled_critic_value

                processed_timesteps.append(
                    processed_timestep(
                        sampled_log_probability=timestep.sampled_log_probability,
                        advantage=advantage,
                        observed_state=timestep.observed_state,
                        action_taken=timestep.action_taken,
                        reward_to_go=value_target,
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
        if self.current_results is None:
            self.current_results = [
                env.reset_result(seed=self.sample_training_seed())
                for env in self.envs
            ]
            self.episode_steps = [0 for _ in self.envs]
        collected_steps = 0

        while collected_steps < config.PPO_ROLLOUT_STEPS:
            states = [
                self.framestack(result.observation, self.episode_steps[i], i)
                for i, result in enumerate(self.current_results)
            ]
            actions, action_log_probs = self.decide_batch(states)
            values = tf.squeeze(self.critic(np.stack(states, axis=0)), axis=1).numpy()

            for i, env in enumerate(self.envs):
                env.start_step(actions[i])

            for i, env in enumerate(self.envs):
                result = env.finish_step()
                next_value = (
                    0.0
                    if result.terminated
                    else self.estimate_next_value(result.observation, i)
                )
                reward = self.reward_from_transition(result)

                active_trajectories[i].append(
                    timestep(
                        observed_state=states[i],
                        action_taken=actions[i],
                        sampled_log_probability=action_log_probs[i],
                        reward=reward,
                        sampled_critic_value=float(values[i]),
                        passed_pipe=result.passed_pipe,
                        terminated=result.terminated,
                        next_critic_value=next_value,
                    )
                )
                collected_steps += 1

                self.episode_steps[i] += 1
                self.current_results[i] = result
                if result.terminated or self.episode_steps[i] >= config.MAX_NUM_STEPS:
                    trajectories.append(active_trajectories[i])
                    self.record_trajectory_stats(active_trajectories[i], result.terminated)
                    active_trajectories[i] = []
                    self.current_results[i] = env.reset_result(
                        seed=self.sample_training_seed()
                    )
                    self.episode_steps[i] = 0

        for trajectory in active_trajectories:
            if trajectory:
                trajectories.append(trajectory)
                self.record_trajectory_stats(trajectory, False)

        return trajectories

    def collect_vector_trajectories(self):
        trajectories = []
        self.last_trajectory_stats = []
        active_trajectories = [[] for _ in range(self.num_envs)]
        if self.current_results is None:
            self.current_results = self.vector_env.reset_all([
                self.sample_training_seed()
                for _ in range(self.num_envs)
            ])
            self.episode_steps = [0 for _ in range(self.num_envs)]
        collected_steps = 0

        while collected_steps < config.PPO_ROLLOUT_STEPS:
            states = [
                self.framestack(result.observation, self.episode_steps[i], i)
                for i, result in enumerate(self.current_results)
            ]
            actions, action_log_probs = self.decide_batch(states)
            values = tf.squeeze(self.critic(np.stack(states, axis=0)), axis=1).numpy()
            next_results = self.vector_env.step_batch(actions)
            next_values = self.estimate_next_values(next_results)

            for i, result in enumerate(next_results):
                reward = self.reward_from_transition(result)

                active_trajectories[i].append(
                    timestep(
                        observed_state=states[i],
                        action_taken=actions[i],
                        sampled_log_probability=action_log_probs[i],
                        reward=reward,
                        sampled_critic_value=float(values[i]),
                        passed_pipe=result.passed_pipe,
                        terminated=result.terminated,
                        next_critic_value=float(next_values[i]),
                    )
                )
                collected_steps += 1

                self.episode_steps[i] += 1
                self.current_results[i] = result
                if result.terminated or self.episode_steps[i] >= config.MAX_NUM_STEPS:
                    trajectories.append(active_trajectories[i])
                    self.record_trajectory_stats(active_trajectories[i], result.terminated)
                    active_trajectories[i] = []
                    self.current_results[i] = self.vector_env.reset_one(
                        i,
                        seed=self.sample_training_seed(),
                    )
                    self.episode_steps[i] = 0

        for trajectory in active_trajectories:
            if trajectory:
                trajectories.append(trajectory)
                self.record_trajectory_stats(trajectory, False)

        return trajectories

    def record_trajectory_stats(self, trajectory, terminated):
        self.last_trajectory_stats.append({
            "length": len(trajectory),
            "reward": sum(timestep.reward for timestep in trajectory),
            "pipes": sum(timestep.passed_pipe for timestep in trajectory),
            "terminated": terminated,
        })

    @staticmethod
    def reward_from_transition(result):
        if result.terminated:
            base_reward = config.REWARD_DIE
        elif result.passed_pipe:
            base_reward = config.REWARD_PASSED_PIPE
        else:
            base_reward = config.REWARD_STD

        return base_reward

    def sample_training_seed(self):
        return int(self.training_seed_rng.uniform(
            shape=(),
            minval=config.TRAIN_SEED_MIN,
            maxval=config.TRAIN_SEED_MAX + 1,
            dtype=tf.int64,
        ).numpy())

    def shuffled_indices(self, size):
        seed = self.next_stateless_seed(self.shuffle_rng)
        return tf.random.experimental.stateless_shuffle(tf.range(size), seed).numpy()

    @staticmethod
    def next_stateless_seed(generator):
        return generator.uniform(
            shape=(2,),
            minval=0,
            maxval=2**31 - 1,
            dtype=tf.int32,
        )

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

    def estimate_next_values(self, results):
        states = [
            self.peek_framestack(result.observation, i)
            for i, result in enumerate(results)
        ]
        values = tf.squeeze(self.critic(np.stack(states, axis=0)), axis=1).numpy()
        return [
            0.0 if result.terminated else float(values[i])
            for i, result in enumerate(results)
        ]

    def estimate_next_value(self, pixels, env_index=0):
        state = self.peek_framestack(pixels, env_index)
        return float(self.critic(state[np.newaxis, ...])[0, 0].numpy())

    def peek_framestack(self, pixels, env_index=0):
        frame = np.frombuffer(pixels, dtype=np.uint8).reshape(
            config.OBS_HEIGHT,
            config.OBS_WIDTH,
            config.OBS_CHANNELS,
        )
        buffer = self.framestack_buffers[env_index]
        if not buffer:
            frames = [frame] * config.FRAME_STACK
        else:
            frames = buffer[1:] + [frame]
        return np.concatenate(frames, axis=-1)

    def decide(self, state):
        logits = self.actor(state[np.newaxis, ...])
        log_probabilities = tf.nn.log_softmax(logits)[0]
        seed = self.next_stateless_seed(self.action_rng)
        action = int(tf.random.stateless_categorical(logits, 1, seed)[0, 0].numpy())
        return action, float(log_probabilities[action].numpy())

    def decide_deterministic(self, state):
        logits = self.actor(state[np.newaxis, ...])
        action = tf.argmax(logits[0], axis=0)
        return int(action.numpy())

    def decide_batch(self, states):
        logits = self.actor(np.stack(states, axis=0))
        log_probabilities = tf.nn.log_softmax(logits)
        seed = self.next_stateless_seed(self.action_rng)
        actions = tf.cast(
            tf.squeeze(tf.random.stateless_categorical(logits, 1, seed), axis=1),
            tf.int32,
        )
        indices = tf.stack([tf.range(tf.shape(actions)[0]), actions], axis=1)
        action_log_probs = tf.gather_nd(log_probabilities, indices)
        return actions.numpy().astype(int).tolist(), action_log_probs.numpy().astype(float).tolist()

    def evaluate_policy(
        self,
        episodes=None,
        max_steps=None,
        env_backend="cpp_vector",
        deterministic=True,
        target_score=None,
        seed_start=None,
    ):
        episodes = episodes or config.VALIDATION_EPISODES
        max_steps = max_steps or config.VALIDATION_MAX_STEPS
        target_score = config.VALIDATION_TARGET_SCORE if target_score is None else target_score
        seed_start = config.VALIDATION_SEED_START if seed_start is None else seed_start

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
                seed_start,
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
                seed_start,
            )
        if env_backend == "subprocess":
            return self._evaluate_subprocess_policy(
                episodes,
                max_steps,
                deterministic,
                target_score,
                seed_start,
            )
        raise ValueError(f"Unsupported validation env_backend: {env_backend!r}")

    def validate_sim_mismatch(
        self,
        episodes=None,
        max_steps=None,
        deterministic=True,
        target_score=None,
        seed_start=None,
    ):
        real_stats = self.evaluate_policy(
            episodes=episodes,
            max_steps=max_steps,
            env_backend="cpp_vector",
            deterministic=deterministic,
            target_score=target_score,
            seed_start=seed_start,
        )
        fast_stats = self.evaluate_policy(
            episodes=episodes,
            max_steps=max_steps,
            env_backend="fast",
            deterministic=deterministic,
            target_score=target_score,
            seed_start=seed_start,
        )
        return {
            "real": real_stats,
            "fast": fast_stats,
            "score_gap": fast_stats["avg_score"] - real_stats["avg_score"],
            "step_gap": fast_stats["avg_steps"] - real_stats["avg_steps"],
        }

    def _evaluate_vector_policy(
        self,
        env,
        episodes,
        max_steps,
        deterministic,
        target_score,
        seed_start,
    ):
        try:
            buffers = [[] for _ in range(episodes)]
            results = env.reset_all([seed_start + i for i in range(episodes)])
            finished = [False for _ in range(episodes)]
            steps = [0 for _ in range(episodes)]
            scores = [0 for _ in range(episodes)]

            while not all(finished):
                active_indices = [i for i, done in enumerate(finished) if not done]
                states = [
                    self.stack_frame_with_buffer(results[i].observation, steps[i], buffers[i])
                    for i in active_indices
                ]
                if deterministic:
                    actions = self.decide_batch_deterministic(states)
                else:
                    actions = self.decide_batch(states)[0]
                batch_actions = [0 for _ in range(episodes)]
                for index, action in zip(active_indices, actions):
                    batch_actions[index] = action

                next_results = env.step_batch(batch_actions)
                for index in active_indices:
                    result = next_results[index]
                    results[index] = result
                    steps[index] += 1
                    scores[index] = result.score
                    if (
                        result.terminated
                        or steps[index] >= max_steps
                        or result.score >= target_score
                    ):
                        finished[index] = True

            return self._summarize_evaluation(scores, steps, target_score)
        finally:
            env.close()

    def _evaluate_subprocess_policy(
        self,
        episodes,
        max_steps,
        deterministic,
        target_score,
        seed_start,
    ):
        scores = []
        steps = []
        env = FlappyEnv(width=config.OBS_WIDTH, height=config.OBS_HEIGHT)
        try:
            for episode in range(episodes):
                buffer = []
                result = env.reset_result(seed=seed_start + episode)
                score = 0
                step_count = 0
                for step in range(max_steps):
                    state = self.stack_frame_with_buffer(result.observation, step, buffer)
                    if deterministic:
                        action = self.decide_deterministic(state)
                    else:
                        action = self.decide(state)[0]
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
