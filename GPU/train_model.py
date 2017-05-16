import h5py
from keras.models import Model
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.core import Activation, Dense, Flatten, Dropout
from keras.optimizers import RMSprop, Adam
from keras.regularizers import l2
from keras import backend as K
import os
import sys
from keras.models import model_from_json
import numpy as np

from collections import deque


class MissingFileException(Exception):
    def __init__(self, msg):
        self.msg = msg


class GPU(object):
    __use_experience_replay = True

    experience_buffer = deque()

    ACTIONS = 6

    model_file = "model/model.json"
    weights_file = "model/weights.h5"

    states = None
    actions = None
    rewards = None
    suc_states = None
    terminals = None

    labels = np.array([])

    model = None

    # Load model config and weights from files
    def create_model(self):

        if os.path.isfile(self.model_file):
            with open(self.model_file) as file:
                model_config = file.readline()
                self.model = model_from_json(model_config)
                self.target_model = model_from_json(model_config)
        else:
            raise MissingFileException("Missing file: Model")

        if os.path.isfile(self.weights_file):
            self.model.load_weights(self.weights_file)
        else:
            raise MissingFileException("Missing file: Weights")

    # Load data
    def load_data(self):

        # Load new data
        for new_data_file in os.listdir("data/new_data"):
            if new_data_file.endswith('.h5'):
                print("Adding data from: %s" % new_data_file)
                with h5py.File("data/new_data/" + new_data_file, 'r') as f:
                    if self.states is None:
                        self.states = f['states'][:]
                        self.actions = f['actions'][:]
                        self.rewards = f['rewards'][:]
                        self.suc_states = f['suc_states'][:]
                        self.terminals = f['terminals'][:]
                    else:
                        self.states = np.concatenate(self.states, f['states'][:])
                        self.actions = np.concatenate(self.actions, f['actions'][:])
                        self.rewards = np.concatenate(self.rewards, f['rewards'][:])
                        self.suc_states = np.concatenate(self.suc_states, f['suc_states'][:])
                        self.terminals = np.concatenate(self.terminals, f['terminals'][:])


                # all data will be saved in data.h5, so no need for old data file
                os.remove("data/new_data/" + new_data_file)

        # Load old data
        if self.__use_experience_replay:
            if os.path.isfile("data/data.h5"):
                with h5py.File("data/data.h5", 'r') as f:
                    if self.states is None:
                        self.states = f['states'][:]
                        self.actions = f['actions'][:]
                        self.rewards = f['rewards'][:]
                        self.suc_states = f['suc_states'][:]
                        self.terminals = f['terminals'][:]
                    else:
                        self.states = np.concatenate(self.states, f['states'][:])
                        self.actions = np.concatenate(self.actions, f['actions'][:])
                        self.rewards = np.concatenate(self.rewards, f['rewards'][:])
                        self.suc_states = np.concatenate(self.suc_states, f['suc_states'][:])
                        self.terminals = np.concatenate(self.terminals, f['terminals'][:])

        if not self.states.size or not self.actions.size or not self.rewards.size or not self.suc_states.size or not self.terminals.size:
            raise MissingFileException("Missing file: No data to process")

    # Save experience data to data.h5 file
    def save_data(self):

        mode = 'w'
        if not self.__use_experience_replay and os.path.isfile("data/data.h5"):
            mode = 'a'

        with h5py.File('data/data.h5', mode) as f:
            f.create_dataset('states', data=self.states)
            f.create_dataset('actions', data=self.actions)
            f.create_dataset('rewards', data=self.rewards)
            f.create_dataset('suc_states', data=self.suc_states)
            f.create_dataset('terminals', data=self.terminals)

    def train_model(self, epochs=, batch_size=5, discount_factor=.9):
        if self.model and self.states.size:

            # init experience buffer
            self.experience_buffer = deque([])

            #print(self.states[0], self.states.shape)

            # draw index for sample to avoid copying large amounts of data
            self.experience_buffer.extend(range(self.states.shape[0]))
            #print("Buffer size: ", len(self.experience_buffer))

            self.model.compile(loss='mean_squared_error', optimizer='sgd')
            self.target_model.set_weights(self.model.get_weights())

            for epoch in range(epochs):
                print("Epoch: ", epoch)

                # sample a minibatch
                minibatch = np.random.choice(self.experience_buffer, size=batch_size)

                # inputs are the states
                inputs = np.array([self.states[i] for i in minibatch])  # 30, 20, 20
                batch_targets = self.model.predict(inputs)  # 30, 6

                print("Batch targets Q-Network:\n", batch_targets)

                # get corresponding successor states for minibatch
                batch_suc_states = np.array([self.suc_states[i] for i in minibatch])  # 30, 20, 20
                batch_terminals = np.array([self.terminals[i] for i in minibatch]).astype('bool')  # 30, 1
                batch_rewards = np.array([self.rewards[i] for i in minibatch])  # 30, 1

                print("Batch terminals:\n", batch_terminals)
                print("Batch rewards:\n", batch_rewards)

                # calculate Q-Values of successor states
                Q_suc = self.target_model.predict(batch_suc_states)
                print("Q-Values for successor states:\n", Q_suc)
                # get max_Q values, discount them and set set those values to 0 where state is terminal
                max_Q_suc = np.amax(Q_suc, axis=1) * discount_factor * np.invert(batch_terminals)
                print("max Q-Values for successor states\n", max_Q_suc)
                argmax_Q_suc = np.argmax(Q_suc, axis=1)
                print("argmax Q-Values for successor states\n", argmax_Q_suc)

                batch_targets[range(batch_size), argmax_Q_suc] = max_Q_suc + batch_rewards
                print("final targets:\n", batch_targets)

                loss = self.model.train_on_batch(inputs, batch_targets)


if __name__ == "__main__":

    gpu = GPU()

    try:
        gpu.load_data()
        gpu.save_data()

        print(gpu.states.shape)

        gpu.create_model()

        gpu.train_model()
    except MissingFileException as e:
        print(e.msg)

        # model = create_model(model_file, weights_file)

        # train_model(model, num_episodes=300)
