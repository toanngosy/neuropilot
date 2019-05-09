import gym
import numpy as np

class ObsWrapper(gym.ObservationWrapper):
	def __init__(self,env):
		super(ObsWrapper, self).__init__(env)

	def observation(self, observation):
		return self.target - self.state


class STRewardWrapper(gym.RewardWrapper):
	def __init__(self,env):
		super(STRewardWrapper, self).__init__(env)

	def reward(self,reward):
		return -np.linalg.norm(self.target - self.state )
