import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as Axes3D
import sys

class SimpleTargetEnv2D(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-0.1, high=0.1, shape=(2,), dtype=np.float32)
        self.reset()

    def reset(self):
        self.state = np.random.uniform(-1, 1, size=(2,))
        observation = np.copy(self.state)
        return observation

    def step(self, action):
        self.state = self.state + action
        x, y = self.state
        reward = - (x**2 + y**2) ** 0.5
        done = abs(x) < 0.01 and abs(y) < 0.01
        next_observation = np.copy(self.state)
        print(self.state, reward)
        return next_observation, reward, done, {}

    def render(self, mode='human', close=False):
        print('current state:', self.state)

class SimpleTargetEnv3D(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-0.1, high=0.1, shape=(3,), dtype=np.float32)
        self.reset()

    def reset(self):
        self.state = np.random.uniform(-1, 1, size=(3,))
        observation = np.copy(self.state)
        return observation

    def step(self, action):
        self.state = self.state + action
        x, y, z = self.state
        reward = - (x**2 + y**2 + z**2) ** 0.5
        done = abs(x) < 0.01 and abs(y) < 0.01
        next_observation = np.copy(self.state)
        print(self.state, reward)
        return next_observation, reward, done, {}

    def render(self, mode='human', close=False):
        print('current state:', self.state)
