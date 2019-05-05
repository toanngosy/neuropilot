import gym
import quadrotorenv
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2, TRPO
from utils.wrappers.wrappers import *
import tensorflow as tf
import numpy as np
policy_kwargs = {
                'act_fun': tf.nn.relu,
                'net_arch': [32, 32, dict(vf=[32], pi=[32])]
                }

env = TargetObservationWrapper(gym.make('QuadRotorEnv-v1'))

env = DummyVecEnv([lambda: env])

model = TRPO('MlpPolicy', env=env,
     verbose=False, gamma=0.90, lam=0.95, cg_iters=10, vf_stepsize=0.003, timesteps_per_batch=1000,
     max_kl=0.3, entcoeff=0.0, cg_damping=0.01, vf_iters=3,
     policy_kwargs=policy_kwargs, tensorboard_log="./tensorboard_log/")

model.learn(1000000)
model.save("TRPO_Pilot")
"""PPO2('MlpPolicy', env=env, learning_rate=0.003,
     policy_kwargs=policy_kwargs, tensorboard_log="./tensorboard_log/").learn(100000000)"""
