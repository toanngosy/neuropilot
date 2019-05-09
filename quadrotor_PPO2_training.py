import gym
import envs.quadrotorenv
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from quadrotor_wrapper import QuadrotorObsWrapper , QuadrotorRewardWrapper


# multiprocess environment

env = DummyVecEnv([lambda: QuadrotorRewardWrapper(QuadrotorObsWrapper(gym.make('QuadRotorEnv-v0')))])
print("Start training")
model = PPO2(MlpPolicy, env, verbose=0, tensorboard_log="./log_quadrotor/")
model.learn(total_timesteps=700000)
model.save("ppo2_quad")
print("Training completed")
