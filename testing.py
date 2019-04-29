import gym
import quadrotorenv


class CustomRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super(CustomRewardWrapper, self).__init__(env)

    def reward(self, reward):
        return 1


env = CustomRewardWrapper(gym.make('QuadRotorEnv-v0'))
state = env.reset()

done = False
i = 0
while not done:
    i += 1
    obs, reward, done, _ = env.step(env.action_space.sample())
    print(reward, end=" ")
