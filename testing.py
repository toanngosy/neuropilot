import gym
import quadrotorenv
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2, TRPO
import tensorflow as tf
import numpy as np
policy_kwargs = {
                'act_fun': tf.nn.relu,
                'net_arch': [32, 32, dict(vf=[32], pi=[32])]
                }

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

        # TODO: fix observation_space bound - @nimbus state[]
        self.observation_space = gym.spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

    def observation(self, observation):
        return np.concatenate([self.target, observation])


env = TargetObservationWrapper(gym.make('QuadRotorEnv-v0'))

TRPO('MlpPolicy', env, \
     verbose=False, policy_kwargs=policy_kwargs, tensorboard_log="./tensorboard_log/").learn(10000000)
