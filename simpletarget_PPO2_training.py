import gym
import envs.simpletargetenv
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2





# multiprocess environment

env = DummyVecEnv([lambda: gym.make('SimpleTargetEnv2D-v0')])

model = PPO2(MlpPolicy, env, learning_rate = 0.1, cliprange = 0.1, verbose=1)
model.learn(total_timesteps=1000000)
model.save("ppo2_simpletarget")