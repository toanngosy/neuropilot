import gym
import envs.quadrotorenv
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from quadrotor_wrapper import QuadrotorObsWrapper , QuadrotorRewardWrapper


# multiprocess environment

env = DummyVecEnv([lambda: QuadrotorRewardWrapper(QuadrotorObsWrapper(gym.make('QuadRotorEnv-v0')))])


model = PPO2.load("ppo2_quad")


for i in range(40):
    obs = env.reset()

    done = False

    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)

        env.render()

