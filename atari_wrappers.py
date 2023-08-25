import gym
import numpy as np
import collections

class RenderWrapper(gym.Wrapper):
    def init(self, env=None):
        super(RenderWrapper, self).__init__(env)
        self.last_obs = None

    @property
    def render_mode(self):
        return 'rgb_array'
    
    def reset(self):
        obs = self.env.reset()
        self.last_obs = obs
        return obs
    
    def step(self, action):
        step_tuple = super().step(action)
        self.last_obs = step_tuple[0]
        return step_tuple

    def render(self, *args, **kwargs):
        return self.last_obs

class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps, dtype=np.float32):
        super(BufferWrapper, self).__init__(env)
        self.dtype = dtype
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(old_space.low.repeat(n_steps, axis=0),
                                                old_space.high.repeat(n_steps, axis=0), dtype=dtype)

    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        return self.observation(self.env.reset())

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer
    
class ImageToPyTorch(gym.ObservationWrapper):
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(old_shape[-1], old_shape[0], old_shape[1]),
                                                dtype=np.uint8)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)