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

states = np.array([])
actions = np.array([])
rewards = np.array([])
suc_states = np.array([])
terminals = np.array([])


# def train_model(model, num_episodes, states, targets):
#
#     gamma = 0.9
#     epsilon = 1.0
#
#     for i in range(num_episodes):
#         status = 1
#         state, progress = create_init_state(wire_arr)
#         while status == 1:
#             state_img = state_to_image(state)
#             qval = model.predict(state_img, batch_size=1)
#             if np.random.random(1) < epsilon:  # choose random action
#                 action = np.random.randint(0, 6)
#             else:
#                 action = np.argmax(qval)
#             new_state = make_move(state, action)
#             reward = get_reward(new_state, state, progress)
#             new_state_img = state_to_image(new_state)
#             newQ = model.predict(new_state_img, batch_size=1)
#             maxQ = np.max(newQ)
#             y = np.zeros((1, 6))
#             if reward not in [-10]:
#                 update = (reward + (gamma*maxQ))
#             else:
#                 update = reward
#             y[0][action] = update  # target output
#             if i % 50 == 0:
#                 model.fit(state_img, y, batch_size=1, nb_epoch=1, verbose=0)
#             state = new_state
#             if reward in [-10]:
#                 status = 0
#         if epsilon > 0.1:
#             epsilon -= (1/num_episodes)


def create_model(model_file, weights_file):
    if os.path.isfile(model_file):
        with open(model_file) as json_string:
            model = model_from_json(json_string)
    else:
        print("Could not find model file at: %s" % model_file)
        return None

    if os.path.isfile(weights_file):
        model.load_weights(weights_file)
    else:
        print("Could not find weights file at: %s" % weights_file)
        return None
    return model

def load_data():
    global states, actions, rewards, suc_states, terminals

    if os.path.isfile("data/data.h5"):
        with h5py.File("data/data.h5", 'r') as f_data:
            states = f_data['states'][:]
            actions = f_data['actions'][:]
            rewards = f_data['rewards'][:]
            suc_states = f_data['suc_states'][:]
            terminals = f_data['terminals'][:]

def add_new_data():
    global states, actions, rewards, suc_states, terminals

    for new_data_file in os.listdir("data/new_data"):
        if new_data_file.endswith('.h5'):
            print("Adding data from: %s" % new_data_file)
            with h5py.File("data/new_data/" + new_data_file, 'r') as f:
                states = f['states'][:]
                actions = f['actions'][:]
                rewards = f['rewards'][:]
                suc_states = f['suc_states'][:]
                terminals = f['terminals'][:]

            # all data will be saved in data.h5, so no need for old data file
            os.remove("data/new_data/" + new_data_file)

def save_data():
    global states, actions, rewards, suc_states, terminals
    with h5py.File('data/data.h5', 'w') as f:
        f.create_dataset('states', data=states)
        f.create_dataset('actions', data=actions)
        f.create_dataset('rewards', data=rewards)
        f.create_dataset('suc_states', data=suc_states)
        f.create_dataset('terminals', data=terminals)

if __name__ == "__main__":

    #model_file = sys.argv[1]
    #weights_file = sys.argv[2]

    load_data()
    print(states.shape, actions.shape)

    add_new_data()
    print(states.shape, actions.shape)

    save_data()

    #model = create_model(model_file, weights_file)



    #train_model(model, num_episodes=300)

