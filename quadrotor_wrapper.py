import gym
import numpy as np

'''
Observation wrapper is made according to this paper:
https://arxiv.org/abs/1707.05110
18-element in observation space:

- 3 relative linear position ( Target linear position - Drone's linear position)
- 3 linear velocity
- 9 element of rotation matrix 
- 3 angular velocity 
'''

class QuadrotorObsWrapper(gym.ObservationWrapper):
    def __init__(self,env):
        super(QuadrotorObsWrapper,self).__init__(env)
        obs_high = np.array([
                   np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max,
                   np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max,
                   np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max,
                   np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max,
                   np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max,
                   np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max])
        obs_low = np.array([
                   np.finfo(np.float32).min, np.finfo(np.float32).min, np.finfo(np.float32).min,
                   np.finfo(np.float32).min, np.finfo(np.float32).min, np.finfo(np.float32).min,
                   np.finfo(np.float32).min, np.finfo(np.float32).min, np.finfo(np.float32).min,
                   np.finfo(np.float32).min, np.finfo(np.float32).min, np.finfo(np.float32).min,
                   np.finfo(np.float32).min, np.finfo(np.float32).min, np.finfo(np.float32).min,
                   np.finfo(np.float32).min, np.finfo(np.float32).min, np.finfo(np.float32).min])
		
		self.observation_space = gym.spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

	def observation(self,observation):
		obs = np.zeros((18,1))
		# Relative linear postion
		obs[0:3] = self.target - self.state[0:3]
		# Linear velocity 
		obs[3:6] = self.state[3:6]
		# 9-element of rotation matrix 
		R = self._rotation_matrix(self.state[6:9])
		obs[6:15] = R.ravel()
		# Angular velocity
		obs[15:18] = self.state[9:12]
		return obs

'''
Reward wrapper is made according to this paper:
https://arxiv.org/abs/1707.05110
'''



class QuadrotorRewardWrapper(gym.RewardWrapper):
	def __init__(self,env):
		super(STRewardWrapper, self).__init__(env)

	def reward(self,reward):

		distance_cost = 4e-3*np.linalg.norm(self.target - self.state[0:3])
		velocity_cost = 5e-4*np.linalg.norm(self.state[3:6])
		pose_cost     = 2e-4*np.linalg.norm(self.state[6:9])
		angular_vel_cost   = 3e-4*np.linalg.norm(self.state[9:12])
		total_cost = distance_cost + velocity_cost + pose_cost + angular_vel_cost

		return -total_cost