import gym
import envs.simpletargetenv
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from simpletarget_wrapper import ObsWrapper , STRewardWrapper





# multiprocess environment

env = DummyVecEnv([lambda: STRewardWrapper(ObsWrapper(gym.make('SimpleTargetEnv2D-v0')))])
print("Start training")
model = PPO2(MlpPolicy, env, learning_rate = 0.1, cliprange = 0.01, verbose=0)
model.learn(total_timesteps=200000)
model.save("ppo2_simpletarget")
print("Training completed")
