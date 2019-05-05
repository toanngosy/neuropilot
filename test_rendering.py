import gym
import envs.simpletargetenv


env = gym.make('SimpleTargetEnv2D-v0')
env.reset()
env.render()