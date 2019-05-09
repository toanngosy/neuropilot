import gym
import envs.simpletargetenv
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from simpletarget_wrapper import ObsWrapper , STRewardWrapper



env = DummyVecEnv([lambda: STRewardWrapper(ObsWrapper(gym.make('SimpleTargetEnv3D-v0')))])


model = PPO2.load("ppo2_simpletarget")


for i in range(40):
    obs = env.reset()

    done = False

    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)

        env.render()

