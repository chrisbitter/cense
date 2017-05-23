import h5py
from keras.models import Model
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.core import Activation, Dense, Flatten, Dropout
from keras.optimizers import RMSprop, Adam
from keras.regularizers import l2
from keras import backend as K
import sys
from keras.models import model_from_json
import numpy as np

from keras.models import Sequential, Model
from keras.layers import Input, Dense, Concatenate, Reshape, Conv2D, MaxPooling2D, Dropout, Flatten, RepeatVector, merge, Activation, Lambda
from keras.layers.merge import Average, Add
import keras.backend as K

from collections import deque

# disables source compilation warnings
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class MissingFileException(Exception):
    def __init__(self, msg):
        self.msg = msg

# v1
def model_dueling(input_shape, output_dim):
    # Common Layers
    input_layer = Input(shape=input_shape)
    common_layer = Reshape(input_shape + (1,))(input_layer)
    common_layer = Conv2D(30, (5, 5), activation="relu")(common_layer)
    # common_layer = MaxPooling2D(pool_size=(2, 2))(common_layer)
    common_layer = Conv2D(15, (3, 3), activation="relu")(common_layer)
    # common_layer = MaxPooling2D(pool_size=(2, 2))(common_layer)
    common_layer = Flatten()(common_layer)

    #adv_layer = Dropout(0.2)(common_layer)
    adv_layer = Dense(128, activation="relu")(common_layer)
    adv_layer = Dense(50, activation="relu")(adv_layer)
    adv_layer = Dense(output_dim, activation="tanh")(adv_layer)

    #val_layer = Dropout(0.2)(common_layer)
    val_layer = Dense(128, activation="relu")(common_layer)
    val_layer = Dense(50, activation="relu")(val_layer)
    val_layer = Dense(1, activation="linear")(val_layer)
    val_layer = RepeatVector(output_dim)(val_layer)
    val_layer = Flatten()(val_layer)
    # q = v + a - mean(a, reduction_indices=1, keep_dims=True)

    #q_layer = val_layer + adv_layer - reduce_mean(adv_layer, keep_dims=True)

    q_layer = merge(inputs=[adv_layer, val_layer], mode=lambda x: x[1] + x[0] - K.mean(x[0], keepdims=True),
                        output_shape=lambda x: x[0])
    #q_layer = Activation(activation="tanh")(q_layer)

    model = Model(inputs=[input_layer], outputs=[q_layer])

    model.compile(loss='mse',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model

class GPU(object):
    __use_experience_replay = True

    experience_buffer = deque()

    ACTIONS = 6

    root_folder = "/home/chris/Thesis/GPU/"

    model_file = root_folder + "model/model.json"
    weights_file = root_folder + "model/weights.h5"

    new_data_folder = root_folder + "data/new_data/"
    data_file = root_folder + "data/data.h5"

    states = None
    actions = None
    rewards = None
    suc_states = None
    terminals = None

    model = None
    target_model = None

    # Load [model config and] weights from files
    def create_model(self):

        #if os.path.isfile(self.model_file):
        #    with open(self.model_file) as file:
        #        model_config = file.readline()
        #        self.model = model_from_json(model_config)
        #        self.target_model = model_from_json(model_config)
        #else:
        #    raise MissingFileException("Missing file: Model")

        self.model = model_dueling((50,50),6)
        self.target_model = model_dueling((50, 50), 6)

        if os.path.isfile(self.weights_file):
            self.model.load_weights(self.weights_file)
        else:
            raise MissingFileException("Missing file: Weights")

    # Load data
    def load_data(self):

        # Load new data
        for new_data_file in os.listdir(self.new_data_folder):
            if new_data_file.endswith('.h5'):
                # print("Adding data from: %s" % new_data_file)
                with h5py.File(self.new_data_folder + new_data_file, 'r') as f:
                    if self.states is None:
                        self.states = f['states'][:]
                        self.actions = f['actions'][:]
                        self.rewards = f['rewards'][:]
                        self.suc_states = f['suc_states'][:]
                        self.terminals = f['terminals'][:]
                    else:
                        self.states = np.concatenate([self.states, f['states'][:]])
                        self.actions = np.concatenate([self.actions, f['actions'][:]])
                        self.rewards = np.concatenate([self.rewards, f['rewards'][:]])
                        self.suc_states = np.concatenate([self.suc_states, f['suc_states'][:]])
                        self.terminals = np.concatenate([self.terminals, f['terminals'][:]])


                # all data will be saved in data.h5, so no need for old data file
                os.remove(self.new_data_folder + new_data_file)

        # Load old data
        if self.__use_experience_replay:
            if os.path.isfile(self.data_file):
                with h5py.File(self.data_file, 'r') as f:
                    if self.states is None:
                        self.states = f['states'][:]
                        self.actions = f['actions'][:]
                        self.rewards = f['rewards'][:]
                        self.suc_states = f['suc_states'][:]
                        self.terminals = f['terminals'][:]
                    else:
                        self.states = np.concatenate([self.states, f['states'][:]])
                        self.actions = np.concatenate([self.actions, f['actions'][:]])
                        self.rewards = np.concatenate([self.rewards, f['rewards'][:]])
                        self.suc_states = np.concatenate([self.suc_states, f['suc_states'][:]])
                        self.terminals = np.concatenate([self.terminals, f['terminals'][:]])

        if self.states is None or not self.states.size:
            raise MissingFileException("Missing file: No data to process")
            
    # Save experience data to data.h5 file
    def save_data(self):

        mode = 'w'
        if not self.__use_experience_replay and os.path.isfile(self.data_file):
            mode = 'a'

        with h5py.File(self.data_file, mode) as f:
            f.create_dataset('states', data=self.states)
            f.create_dataset('actions', data=self.actions)
            f.create_dataset('rewards', data=self.rewards)
            f.create_dataset('suc_states', data=self.suc_states)
            f.create_dataset('terminals', data=self.terminals)

    def train_model(self, epochs=2, batch_size=3, discount_factor=.9):
        if self.model and self.states.size:

            # init experience buffer
            self.experience_buffer = deque([])

            # print(self.states[0], self.states.shape)

            # draw index for sample to avoid copying large amounts of data
            self.experience_buffer.extend(range(self.states.shape[0]))
            # print("Buffer size: ", len(self.experience_buffer))

            self.model.compile(loss='mean_squared_error', optimizer='sgd')
            self.target_model.set_weights(self.model.get_weights())

            for epoch in range(epochs):
                # print("Epoch: ", epoch)

                # sample a minibatch
                minibatch = np.random.choice(self.experience_buffer, size=batch_size)

                # inputs are the states
                inputs = np.array([self.states[i] for i in minibatch])  # 30, 20, 20
                batch_targets = self.model.predict_on_batch(inputs)  # 30, 6

                # print("Batch targets Q-Network:\n", batch_targets)

                # get corresponding successor states for minibatch
                batch_suc_states = np.array([self.suc_states[i] for i in minibatch])  # 30, 20, 20
                batch_terminals = np.array([self.terminals[i] for i in minibatch]).astype('bool')  # 30, 1
                batch_rewards = np.array([self.rewards[i] for i in minibatch])  # 30, 1

                # print("Batch terminals:\n", batch_terminals)
                # print("Batch rewards:\n", batch_rewards)

                # calculate Q-Values of successor states
                Q_suc = self.target_model.predict(batch_suc_states)
                # print("Q-Values for successor states:\n", Q_suc)
                # get max_Q values, discount them and set set those values to 0 where state is terminal
                max_Q_suc = np.amax(Q_suc, axis=1) * discount_factor * np.invert(batch_terminals)
                # print("max Q-Values for successor states (discounted)\n", max_Q_suc)
                argmax_Q_suc = np.argmax(Q_suc, axis=1)
                # print("argmax Q-Values for successor states\n", argmax_Q_suc)

                batch_targets[range(batch_size), argmax_Q_suc] = max_Q_suc + batch_rewards
                # print("final targets:\n", batch_targets)

                loss = self.model.train_on_batch(inputs, batch_targets)


if __name__ == "__main__":

    gpu = GPU()

    try:
        gpu.load_data()
        gpu.save_data()

        # print(gpu.states.shape)

        gpu.create_model()

        gpu.train_model()

        print("done")

    except MissingFileException as e:
        print(e.msg)

        # model = create_model(model_file, weights_file)

        # train_model(model, num_episodes=300)
