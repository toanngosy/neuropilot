import gym
import quadrotorenv

env = gym.make('QuadRotorEnv-v0')
state = env.reset()
for i in range(100):
    obs, _, _, _ = env.step(env.action_space.sample())
    print(i, obs)
