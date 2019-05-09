import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as Axes3D
import sys


class SimpleTargetEnv2D(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-0.1, high=0.1, shape=(2,), dtype=np.float32)

        #defind enviroment bound

        self.min_x = 0
        self.max_x = 10
        self.min_y = 0
        self.max_y = 10

        self.target = np.zeros(self.observation_space.shape)
        self.target[0] = 8
        self.target[1] = 8

        self.state = np.zeros(self.observation_space.shape)
        self.state[0] = 1
        self.state[1] = 1
        self.viewer = None
        self.verbal = False

    def reset(self):
        #self.state[0] = np.random.uniform(self.min_x, self.max_x)
        #self.state[1] = np.random.uniform(self.min_y, self.max_y)
        self.target[0] = np.random.uniform(self.min_x, self.max_x)
        self.target[1] = np.random.uniform(self.min_y, self.max_y)

        observation = np.copy(self.state)
        return observation

    def step(self, action):
        self.state = self.state + action
        reward = - np.linalg.norm(self.target - self.state)

        done = self._reached()
        next_observation = np.copy(self.state)
        if self.verbal:
            print(self.state, reward)
        return next_observation, reward, done, {}

    def _reached(self):
        distance = np.linalg.norm(self.target - self.state)

        return distance < 1

    def _crashed(self):
        return (self.state[0] < self.min_x) or \
               (self.state[0] > self.max_x) or \
               (self.state[1] < self.min_y) or \
               (self.state[1] > self.max_y)    \

    def reward(self):
        #reward = - np.linalg.norm(self.target - self.state)
        reward = 0

        if self._reached():
            reward = 100

        if self._crashed():
            reward = - 100

        return reward

    def render(self, mode='human', close=False):
        if self.viewer is None:
            #initialize renderings
            self.viewer, self.ax = plt.subplots()
            self.ax.set_xlim([self.min_x-5, self.max_x+5])
            self.ax.set_xlabel('X')
            self.ax.set_ylim([self.min_y-5, self.max_y+5])
            self.ax.set_ylabel('Y')
            self.agentplot, = self.ax.plot([], [], marker='o', color='blue', markersize=6, antialiased=False)
            #self.target, = self.ax.plot([], [], marker='o', color='blue', markersize=6, antialiased=False)
            self.targetplot, = self.ax.plot([], [],  marker='o', color='red', markersize=6, antialiased=False)

        self.agentplot.set_data([self.state[0]], [self.state[1]])
        self.targetplot.set_data([self.target[0]], [self.target[1]])
        plt.pause(1e-12)

    def close(self):
        if self.viewer:
            plt.close()
            self.viewer = None


class SimpleTargetEnv3D(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-0.1, high=0.1, shape=(3,), dtype=np.float32)

        #defind enviroment bound

        self.min_x = 0
        self.max_x = 10
        self.min_y = 0
        self.max_y = 10
        self.min_z = 0
        self.max_z = 10

        self.target = np.zeros(self.observation_space.shape)
        self.target[0] = 8
        self.target[1] = 8
        self.target[2] = 8

        self.state = np.zeros(self.observation_space.shape)
        self.state[0] = 1
        self.state[1] = 1
        self.state[2] = 1
        self.viewer = None
        self.verbal = False

    def reset(self):
        self.state[0] = np.random.uniform(self.min_x, self.max_x)
        self.state[1] = np.random.uniform(self.min_y, self.max_y)
        self.state[2] = np.random.uniform(self.min_z, self.max_z)
        self.target[0] = np.random.uniform(self.min_x, self.max_x)
        self.target[1] = np.random.uniform(self.min_y, self.max_y)
        self.target[2] = np.random.uniform(self.min_z, self.max_z)

        observation = np.copy(self.state)
        return observation

    def step(self, action):
        self.state = self.state + action
        x, y, z = self.state
        reward = - np.linalg.norm(self.target - self.state)
        done = self._reached()
        next_observation = np.copy(self.state)
        if self.verbal:
            print(self.state, reward)
        return next_observation, reward, done, {}

    def _reached(self):
        distance = np.linalg.norm(self.target - self.state)

        return distance < 1

    def _crashed(self):
        return (self.state[0] < self.min_x) or \
               (self.state[0] > self.max_x) or \
               (self.state[1] < self.min_y) or \
               (self.state[1] > self.max_y) or \
               (self.state[2] < self.min_z) or \
               (self.state[2] > self.max_z)

    def reward(self):
        #reward = - np.linalg.norm(self.target - self.state)
        reward = 0

        if self._reached():
            reward = 100

        if self._crashed():
            reward = - 100

        return reward

    def render(self, mode='human', close=False):
        if self.viewer is None:
            #initialize renderings
            self.viewer, self.ax = plt.subplots(figsize=(30,10), ncols=3, nrows=1)
            self.ax[0].set_xlim([self.min_x-5, self.max_x+5])
            self.ax[0].set_xlabel('X')
            self.ax[0].set_ylim([self.min_y-5, self.max_y+5])
            self.ax[0].set_ylabel('Y')
            self.agentplotxy, = self.ax[0].plot([], [], marker='o', color='blue', markersize=6, antialiased=False)
            self.targetplotxy, = self.ax[0].plot([], [],  marker='o', color='red', markersize=6, antialiased=False)

            self.ax[1].set_xlim([self.min_x-5, self.max_x+5])
            self.ax[1].set_xlabel('X')
            self.ax[1].set_ylim([self.min_z-5, self.max_z+5])
            self.ax[1].set_ylabel('Z')
            self.agentplotxz, = self.ax[1].plot([], [], marker='o', color='blue', markersize=6, antialiased=False)
            self.targetplotxz, = self.ax[1].plot([], [],  marker='o', color='red', markersize=6, antialiased=False)

            self.ax[2].set_xlim([self.min_y-5, self.max_y+5])
            self.ax[2].set_xlabel('Y')
            self.ax[2].set_ylim([self.min_z-5, self.max_z+5])
            self.ax[2].set_ylabel('Z')
            self.agentplotyz, = self.ax[2].plot([], [], marker='o', color='blue', markersize=6, antialiased=False)
            self.targetplotyz, = self.ax[2].plot([], [],  marker='o', color='red', markersize=6, antialiased=False)

        self.agentplotxy.set_data([self.state[0]], [self.state[1]])
        self.targetplotxy.set_data([self.target[0]], [self.target[1]])

        self.agentplotxz.set_data([self.state[0]], [self.state[2]])
        self.targetplotxz.set_data([self.target[0]], [self.target[2]])

        self.agentplotyz.set_data([self.state[1]], [self.state[2]])
        self.targetplotyz.set_data([self.target[1]], [self.target[2]])
        plt.pause(1e-12)

    def close(self):
        if self.viewer:
            plt.close()
            self.viewer = None


