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
import math
import json

from keras.models import Sequential, Model
from keras.layers import Input, Dense, Concatenate, Reshape, Conv2D, MaxPooling2D, Dropout, Flatten, RepeatVector, \
    merge, Activation, Lambda
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
def model_acceleration_q(image_input_shape, velocity_input_shape,  output_dim):

    num_outputs = np.prod(output_dim)

    # this part of the network processes the
    img_input = Input(shape=image_input_shape)
    image_layer = Reshape(image_input_shape + (1,))(img_input)
    image_layer = Conv2D(30, (5, 5), activation="relu")(image_layer)
    # image_layer = MaxPooling2D(pool_size=(2, 2))(image_layer)
    image_layer = Conv2D(15, (5, 5), activation="relu")(image_layer)
    # image_layer = MaxPooling2D(pool_size=(2, 2))(image_layer)
    image_layer = Flatten()(image_layer)

    # this part takes care of the velocity input values
    vel_input = Input(shape=velocity_input_shape)
    vel_layer = Reshape(velocity_input_shape + (1,))(vel_input)
    vel_layer = Flatten()(vel_layer)


    # here, the preprocessed image and the velocities are merged into one tensor
    concat_layer = Concatenate()([image_layer, vel_layer])


    # advantage function of actions
    adv_layer = Dense(100, activation="relu")(concat_layer)
    adv_layer = Dense(50, activation="relu")(adv_layer)
    adv_layer = Dense(num_outputs, activation="tanh")(adv_layer)

    # value of state
    val_layer = Dense(100, activation="relu")(concat_layer)
    val_layer = Dense(50, activation="relu")(val_layer)
    val_layer = Dense(1, activation="linear")(val_layer)
    val_layer = RepeatVector(num_outputs)(val_layer)
    val_layer = Flatten()(val_layer)
    # q = v + a - mean(a, reduction_indices=1, keep_dims=True)
    # q_layer = val_layer + adv_layer - reduce_mean(adv_layer, keep_dims=True)

    # merging advantage function and state value
    q_layer = merge(inputs=[adv_layer, val_layer], mode=lambda x: x[1] + x[0] - K.mean(x[0], keepdims=True),
                    output_shape=lambda x: x[0])
    # q_layer = Activation(activation="tanh")(q_layer)

    q_layer = Reshape(output_dim)(q_layer)

    model = Model(inputs=[img_input, vel_input], outputs=[q_layer])

    model.compile(loss='mse',
                  optimizer='adam',
                  metrics=['accuracy'])

    print(model.summary())

    return model


class Training(object):
    experience_buffer = deque()

    STATE_DIMENSIONS = (50, 50)
    VELOCITY_DIMENSIONS = (3,)
    ACTIONS = (3,3)

    root_folder = "/home/useradmin/Dokumente/rm505424/CENSE/Christian/training_data/"

    train_parameters = root_folder + "train_params.json"

    model_file = root_folder + "model/model.json"
    weights_file = root_folder + "model/weights.h5"
    target_weights_file = root_folder + "model/target_weights.h5"

    new_data_folder = root_folder + "data/new_data/"
    data_file = root_folder + "data/data.h5"

    states = None
    velocities = None
    actions = None
    rewards = None
    suc_states = None
    terminals = None

    model = None
    target_model = None

    use_target = False

    def run(self):
        # create models
        with open(self.train_parameters) as json_data:
            self.config = json.load(json_data)

        self.model = model_acceleration_q(self.STATE_DIMENSIONS, self.VELOCITY_DIMENSIONS, self.ACTIONS)
        self.target_model = model_acceleration_q(self.STATE_DIMENSIONS, self.VELOCITY_DIMENSIONS, self.ACTIONS)

        if os.path.isfile(self.weights_file):
            self.model.load_weights(self.weights_file)
        else:
            raise MissingFileException("Missing file: Weights")

        self.use_target = self.config["use_target"]

        if self.use_target:
            if os.path.isfile(self.target_weights_file):
                self.target_model.load_weights(self.target_weights_file)
            else:
                self.target_model.load_weights(self.weights_file)

        # Load data
        # Load old data
        if os.path.isfile(self.data_file):
            with h5py.File(self.data_file, 'r') as f:
                self.states = f['states'][:]
                self.velocities = f['velocities'][:]
                self.actions = f['actions'][:]
                self.rewards = f['rewards'][:]
                self.suc_states = f['suc_states'][:]
                self.suc_velocities = f['suc_velocities'][:]
                self.terminals = f['terminals'][:]

        # Load new data
        for new_data_file in os.listdir(self.new_data_folder):
            if new_data_file.endswith('.h5'):
                # print("Adding data from: %s" % new_data_file)
                with h5py.File(self.new_data_folder + new_data_file, 'r') as f:
                    if self.states is None:
                        self.states = f['states'][:]
                        self.velocities = f['velocities'][:]
                        self.actions = f['actions'][:]
                        self.rewards = f['rewards'][:]
                        self.suc_states = f['suc_states'][:]
                        self.suc_velocities = f['suc_velocities'][:]
                        self.terminals = f['terminals'][:]
                    else:
                        self.states = np.concatenate([self.states, f['states'][:]])
                        self.velocities = np.concatenate([self.velocities, f['velocities'][:]])
                        self.actions = np.concatenate([self.actions, f['actions'][:]])
                        self.rewards = np.concatenate([self.rewards, f['rewards'][:]])
                        self.suc_states = np.concatenate([self.suc_states, f['suc_states'][:]])
                        self.suc_velocities = np.concatenate([self.suc_velocities, f['suc_velocities'][:]])
                        self.terminals = np.concatenate([self.terminals, f['terminals'][:]])

                # all data will be saved in data.h5, so no need for old data file
                os.remove(self.new_data_folder + new_data_file)

        if self.states is None or not self.states.size:
            raise MissingFileException("Missing file: No data to process")

        # Save experience data to data.h5 file
        with h5py.File(self.data_file, 'w') as f:
            f.create_dataset('states', data=self.states)
            f.create_dataset('velocities', data=self.velocities)
            f.create_dataset('actions', data=self.actions)
            f.create_dataset('rewards', data=self.rewards)
            f.create_dataset('suc_states', data=self.suc_states)
            f.create_dataset('suc_velocities', data=self.suc_velocities)
            f.create_dataset('terminals', data=self.terminals)

        # training

        epochs = self.config["epochs"]
        batch_size = self.config["batch_size"]
        discount_factor = self.config["discount_factor"]
        target_update_rate = self.config["target_update_rate"]

        if self.model and self.states.size:
            # init experience buffer
            self.experience_buffer = deque([])

            # draw index for sample to avoid copying large amounts of data
            self.experience_buffer.extend(range(self.states.shape[0]))

            self.model.compile(loss='mean_squared_error', optimizer='sgd')

            for epoch in range(epochs):

                # sample a minibatch
                minibatch = np.random.choice(self.experience_buffer, size=batch_size)

                # inputs are the states
                batch_states = np.array([[self.states[i], self.velocities[i]] for i in minibatch])  # bx[40x40, 3x1]
                batch_targets = self.model.predict_on_batch(batch_states)  # bx3x3

                batch_actions = np.array([self.actions[i] for i in minibatch]) # bx3

                # print("Batch targets Q-Network:\n", batch_targets)

                # get corresponding successor states for minibatch
                batch_suc_states = np.array([[self.suc_states[i], self.suc_velocities[i]] for i in minibatch])  # bx[40x40,3x1]
                batch_terminals = np.array([self.terminals[i] for i in minibatch]).astype('bool')  # bx1
                batch_rewards = np.array([self.rewards[i] for i in minibatch])  # bx1

                # calculate Q-Values of successor states
                if self.use_target:
                    Q_suc = self.target_model.predict(batch_suc_states) # bx3x3
                else:
                    Q_suc = self.model.predict(batch_suc_states) # bx3x3
                # print("Q-Values for successor states:\n", Q_suc)
                # get max_Q values, discount them and set set those values to 0 where state is terminal
                max_Q_suc = np.amax(Q_suc, axis=2) * discount_factor * np.repeat(np.expand_dims(np.invert(batch_terminals),axis=1), 3, axis=1) # bx3

                # print("max Q-Values for successor states (discounted)\n", max_Q_suc)
                #argmax_Q_suc = np.argmax(Q_suc, axis=1)
                # print("argmax Q-Values for successor states\n", argmax_Q_suc)

                for i in range(batch_size):
                    batch_targets[i][range(self.ACTIONS[0]), batch_actions[i]] = max_Q_suc[i] + batch_rewards[i]

                # batch_targets[range(batch_size), batch_actions] = max_Q_suc + batch_rewards
                # print("final targets:\n", batch_targets)

                self.model.train_on_batch(batch_states, batch_targets)

                if self.use_target:
                    # update target network
                    model_weights = self.model.get_weights()
                    target_weights = self.target_model.get_weights()
                    for i in range(len(model_weights)):
                        target_weights[i] = target_update_rate * model_weights[i] + (1 - target_update_rate) * \
                                                                                    target_weights[i]
                    self.target_model.set_weights(target_weights)

        self.model.save_weights(self.weights_file)
        if self.use_target:
            self.target_model.save_weights(self.target_weights_file)


if __name__ == "__main__":

    try:
        Training().run()

    except MissingFileException as e:
        print(e.msg)