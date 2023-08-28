import gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class ModelMetricsCallback(BaseCallback):
    def __init__(self, eval_env: gym.Env, save_path, num_episodes = 10, mean_rewards = [], rewards_stds = [],verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.save_path = save_path
        self.num_episodes = num_episodes
        self.mean_rewards = mean_rewards
        self.rewards_stds = rewards_stds
        
    def _on_step(self) -> bool:
        assert self.parent is not None, "``StopTrainingOnMinimumReward`` callback must be used " "with an ``EventCallback``"

        rewards = []
        for _ in range(self.num_episodes):       
          obs = self.eval_env.reset()
          done = False
          ep_sum_rewards = 0
          while not done:
              action, _ = self.model.predict(obs, deterministic=True)
              obs, reward, done, _ = self.eval_env.step(int(action))
              ep_sum_rewards += reward
        
          rewards.append(ep_sum_rewards)
          
          if (self.verbose >= 2):
              print('Episode done')

        mean_reward = np.mean(rewards)
        rewards_std = np.std(rewards)

        if (self.verbose >= 1):
            print('````````````````````````')
            print(f'mean reward ({self.num_episodes} episode(s)): {mean_reward}')
            print('````````````````````````')

        self.mean_rewards.append(mean_reward)
        self.rewards_stds.append(rewards_std)
        np.savetxt(f'{self.save_path}/mean_rewards.txt', self.mean_rewards)
        np.savetxt(f'{self.save_path}/rewards_stds.txt', self.rewards_stds)

        return True