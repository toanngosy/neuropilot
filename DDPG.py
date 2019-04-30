from keras.layers import Dense, Activation, Flatten, Input, Add, concatenate, Lambda
#from keras.layers.normalization import BatchNormalization, LayerNormalization
from keras.optimizers import Adam, SGD, RMSprop
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from keras.models import Sequential, Model
import tensorflow as tf

import gym
import quadrotorenv
import argparse
from datetime import datetime
from quadtensorboard import QuadTensorBoard

"""parser = argparse.ArgumentParser(description='Train or test neural net motor controller')
parser.add_argument('--step', dest='step', action='store', default=2000)
parser.add_argument('--train', dest='train', action='store_true', default=True)
parser.add_argument('--test', dest='train', action='store_false', default=True)
parser.add_argument('--visualize', dest='visualize', action='store_true', default=False)
args = parser.parse_args()"""

env = gym.make('QuadRotorEnv-v0')
env.reset()

# #### ACTOR / CRITIC #####
input_shape = (1, env.observation_space.shape[0])
action_size = env.action_space.shape[0]
SIZE_HIDDEN_LAYER_ACTOR = 64
LR_ACTOR = 0.001
SIZE_HIDDEN_LAYER_CRITIC = 128
LR_CRITIC = 0.001
REPLAY_BUFFER_SIZE = 100000
THETA = 0
MU = 4500
SIGMA = 0
DISC_FACT = 0.9
TARGET_MODEL_UPDATE = 0.001
BATCH_SIZE = 128
ACTION_REPETITION = 5
N_STEPS_TRAIN = 25000
LOG_INTERVAL = 1000

observation_input = Input(shape=input_shape, name='observation_input')
x = Flatten()(observation_input)
x = Dense(SIZE_HIDDEN_LAYER_ACTOR)(x)
# x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dense(SIZE_HIDDEN_LAYER_ACTOR)(x)
# x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dense(SIZE_HIDDEN_LAYER_ACTOR)(x)
# x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dense(action_size)(x)
x = Activation('sigmoid')(x)
x = Lambda(lambda x: x * 9000)(x)
actor = Model(inputs=observation_input, outputs=x)
opti_actor = Adam(lr=LR_ACTOR)

import numpy as np

#print(actor.predict(np.array(env.observation_space.sample()).reshape((1, input_shape))))
# Critic (Q) ##
action_input = Input(shape=(action_size,), name='action_input')
x = Flatten()(observation_input)
x = concatenate([action_input, x])
x = Dense(SIZE_HIDDEN_LAYER_CRITIC)(x)
# x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dense(SIZE_HIDDEN_LAYER_CRITIC)(x)
# x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dense(SIZE_HIDDEN_LAYER_CRITIC)(x)
# x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dense(1)(x)
x = Activation('linear')(x)
critic = Model(inputs=[action_input, observation_input], outputs=x)
opti_critic = Adam(lr=LR_CRITIC)


# #### SET UP THE AGENT #####
# Initialize Replay Buffer ##
memory = SequentialMemory(limit=REPLAY_BUFFER_SIZE, window_length=1)
# window_length : usefull for Atari game (cb d'images d'affilé on veut analysé (vitesse de la balle, etc..))

# Random process (exploration) ##
random_process = OrnsteinUhlenbeckProcess(theta=THETA, mu=MU, sigma=SIGMA,
                                          size=action_size)

# Paramètres agent DDPG ##
agent = DDPGAgent(nb_actions=action_size, actor=actor, critic=critic,
                  critic_action_input=action_input,
                  memory=memory, random_process=random_process,
                  gamma=DISC_FACT, target_model_update=TARGET_MODEL_UPDATE,
                  batch_size=BATCH_SIZE)

agent.compile(optimizer=[opti_critic, opti_actor])


logdir = "logs/" + datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
quadtensorboard = QuadTensorBoard(log_dir=logdir)

agent.fit(env, nb_steps=N_STEPS_TRAIN,
          verbose=True, log_interval=LOG_INTERVAL, visualize=False,
          callbacks=[quadtensorboard], action_repetition = ACTION_REPETITION)

agent.test(env, nb_episodes=10, visualize=True)
