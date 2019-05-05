import timeit
# TensorBoard to monitor
from keras.callbacks import TensorBoard
import numpy as np
from tensorflow.contrib.training import HParams
import tensorflow as tf

class QuadTensorBoard(TensorBoard):
    """
    Subclassing of tensorboard to log and visualize custom metrics and others
    """
    def __init__(self, *args, **kwargs):

        #self.hyperparams = hyperparams
        self.episode_start = {}
        self.observations = {}
        self.rewards = {}
        self.actions = {}
        self.metrics = {}

        super(QuadTensorBoard, self).__init__(*args, **kwargs)

    def _set_env(self, env):
        self.env = env

    def on_step_begin(self, step, logs):
        pass

    def on_step_end(self, step, logs):
        """ Update statistics of episode after each step """
        episode = logs['episode']
        self.observations[episode].append(logs['observation'])
        self.rewards[episode].append(logs['reward'])
        self.actions[episode].append(logs['action'])
        self.metrics[episode].append(logs['metrics'])

    def on_episode_begin(self, episode, logs):
        self.episode_start_time = timeit.default_timer()
        self.observations[episode] = []
        self.rewards[episode] = []
        self.actions[episode] = []
        self.metrics[episode] = []

    def on_episode_end(self, episode, logs={}):
        new_logs = {}
        training_speed = (timeit.default_timer() - self.episode_start_time)/logs["nb_episode_steps"]
        new_logs.update({"Total Reward per Episode": logs["episode_reward"]})
        new_logs.update({"Training Speed per Episode": training_speed})
        new_logs.update({"Number Steps per Episode": logs["nb_episode_steps"]})

        super(QuadTensorBoard, self).on_epoch_end(episode, new_logs)

        del self.observations[episode]
        del self.rewards[episode]
        del self.actions[episode]
        del self.metrics[episode]

    def on_train_begin(self, logs):
        pass

    def on_train_end(self, logs):
        """hparams = HParams(SIZE_HIDDEN_LAYER_ACTOR=self.hyperparams[0],
                                                  LR_ACTOR=self.hyperparams[1],
                                                  SIZE_HIDDEN_LAYER_CRITIC=self.hyperparams[2],
                                                  LR_CRITIC=self.hyperparams[3],
                                                  DISC_FACT=self.hyperparams[4],
                                                  TARGET_MODEL_UPDATE=self.hyperparams[5],
                                                  BATCH_SIZE=self.hyperparams[6],
                                                  REPLAY_BUFFER_SIZE=self.hyperparams[7],
                                                  THETA=self.hyperparams[8],
                                                  SIGMA=self.hyperparams[9]
                                                  )"""
        pass

