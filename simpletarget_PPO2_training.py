import gym
import envs.simpletargetenv
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from simpletarget_wrapper import ObsWrapper , STRewardWrapper


# multiprocess environment

env = DummyVecEnv([lambda: STRewardWrapper(ObsWrapper(gym.make('SimpleTargetEnv3D-v0')))])
print("Start training")
model = PPO2(MlpPolicy, env, verbose=0, tensorboard_log="./log_simpletarget/")
model.learn(total_timesteps=200000)
model.save("ppo2_simpletarget")
print("Training completed")
