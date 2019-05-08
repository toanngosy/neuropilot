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
        self.state[0] = np.random.uniform(self.min_x, self.max_x)
        self.state[1] = np.random.uniform(self.min_y, self.max_y)
        self.target[0] = np.random.uniform(self.min_x, self.max_x)
        self.target[1] = np.random.uniform(self.min_y, self.max_y)

        observation = np.copy(self.state)
        self.close()
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

        return  distance < 1
        
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
            self.agent, = self.ax.plot([], [], marker='o', color='blue', markersize=6, antialiased=False)
            #self.target, = self.ax.plot([], [], marker='o', color='blue', markersize=6, antialiased=False)
            self.ax.plot([self.target[0]], [self.target[1]],  marker='o', color='red', markersize=6, antialiased=False)


        self.agent.set_data([self.state[0]], [self.state[1]])
        #self.target.set_data([self.target[0]], [self.target[1]])
        plt.pause(1e-12)


    def close(self):
        if self.viewer:
            plt.close()
            self.viewer = None

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
