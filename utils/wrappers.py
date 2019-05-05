import gym
import numpy as np

# DroneEnv observation wrapper
class TargetObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(TargetObservationWrapper, self).__init__(env)
        obs_high = np.array([
                   self.max_x, self.max_y, self.max_z,
                   self.max_x, self.max_y, self.max_z,
                   np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max,
                   np.pi/2, np.pi/2, np.pi/2,
                   np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max])
        obs_low = np.array([
                   self.min_x, self.min_y, self.min_z,
                   self.min_x, self.min_y, self.min_z,
                   np.finfo(np.float32).min, np.finfo(np.float32).min, np.finfo(np.float32).min,
                   -np.pi/2, -np.pi/2, -np.pi/2,
                   np.finfo(np.float32).min, np.finfo(np.float32).min, np.finfo(np.float32).min])

        self.observation_space = gym.spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

    def observation(self, observation):
        return np.concatenate([observation, self.target])
