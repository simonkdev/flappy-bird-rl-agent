from critic import Critic
from actor import Actor
import numpy as np
import random
import config

@dataclass
class processed_timestep:
    sampled_probability: float
    advantage: float
    observed_state: np.ndarray
    action_taken: int
    reward_to_go: float
    sampled_critic_value: float

@dataclass
class timestep:
    observed_state: np.ndarray
    action_taken: int
    sampled_probability: float
    reward: float
    sampled_critic_value: float
    reward_to_go: float

class PPO:
#    name: str = ""

    def __init__(self):
        self.actor: Actor = Actor(num_actions=2)
        self.critic: Critic = Critic()

    def training_epoch(self):
        processed_timesteps = self.sample_run()  # python list, 1D with objects in it
        self.actor_training_run(processed_timesteps)
        self.critic_training_run(processed_timesteps)

    def actor_training_run(self, processed_timesteps):
        timesteps = processed_timesteps
        random.shuffle(timesteps)

        batches = [[] for _ in range(config.PPO_MINIBATCH_SIZE)]
        for i, timestep in enumerate(timesteps):
            batches[i % num_arrays].append(timestep)

        for i in range(config.PPO_EPOCHS):
            for batch in batches:
                self.actor_training_step(batch)

    def actor_training_step(self,batch):
        with tf.GradientTape() as tape:
            ov_obj = self.overall_objective(batch)
            loss = 0 - ov_obj
        self.actor.update(loss, tape)

    def overall_objective(self, batch):
        objectives = np.array([])
        for timestep in batch:
            new_prob = self.actor.call(timestep.observed_state, return_probs=True)[timestep.action_taken]
            ratio = new_prob / timestep.sampled_probability
            a = ratio * timestep.advantage
            b = tf.clamp(ratio,ratio - config.PPO_CLIP_EPSILON, ratio + config.PPO_CLIP_EPSILON)* timestep.advantage
            objective = np.min(np.array([a, b]))
            np.append(objectives, objective)
        avg = np.mean(objectives, axis=0)
        return avg

    def critic_training_run(self, processed_timesteps):
        timesteps = processed_timesteps
        random.shuffle(timesteps)

        batches = [[] for _ in range(config.PPO_MINIBATCH_SIZE)]
        for i, timestep in enumerate(timesteps):
            batches[i % num_arrays].append(timestep)

        for i in range(config.PPO_EPOCHS):
            for batch in batches:
                self.critic_training_step(batch)

    def critic_training_step(self, batch):
        with tf.GradientTape() as tape:
            predictions = np.array([])
            rtgs = np.array([])
            for timestep in batch:
                np.append(predictions, timestep.sampled_critic_value)
                np.append(rtgs, timestep.reward_to_go)
            deltas = np.subtract(rtgs, predictions)
            deltas = np.power(deltas,2)
            loss = np.mean(deltas, axis=0)
        self.critic.update(loss, tape)

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
                      sampled_probability=timestep.sampled_probability,
                      advantage=reward_to_go - timestep.sampled_critic_value,
                      observed_state=timestep.observed_state,
                      action_taken=timestep.action_taken,
                      reward_to_go=reward_to_go,
                      sampled_critic_value=timestep.sampled_critic_value,
                  )
              )

      return processed_timesteps




    def testdump(self):
        observation = np.random.rand(1, 42, 42, 5)*255
        print(self.actor.call(observation))
        print(self.critic.call(observation))

ppotest = PPO()
ppotest.testdump()
