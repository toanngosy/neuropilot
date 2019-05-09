import gym
import envs.quadrotorenv

env = gym.make('QuadRotorEnv-v0')
env.reset()
env.render()