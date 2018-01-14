import h5py
from keras import backend as K

import nn_factory

import time

import numpy as np
import json

from collections import deque

# disables source compilation warnings
import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if len(sys.argv) > 1:
    _ID = sys.argv[1]
else:
    _ID = str(np.random.randint(10000))

STATE_DIMENSIONS = (40, 40)
ACTIONS = 5

root_folder = "/home/useradmin/Dokumente/rm505424/CENSE/Christian/training_data/"

train_parameters = root_folder + "train_params.json"

model_file = root_folder + "model/model.h5"
target_file = root_folder + "model/target.h5"

new_data_folder = root_folder + "data/new_data/"
data_file = root_folder + "data/data.h5"

training_signal = root_folder + "training_signal_" + _ID
alive_signal = root_folder + "alive_signal_" + _ID

with open(alive_signal, 'a'):
    pass

model = nn_factory.model_dueling(STATE_DIMENSIONS, ACTIONS)
target = nn_factory.model_dueling(STATE_DIMENSIONS, ACTIONS)

if os.path.isfile(model_file):
    model.load_weights(model_file)
else:
    raise IOError("Missing file: Model")

if os.path.isfile(target_file):
    target.load_weights(target_file)
else:
    target.set_weights(model.get_weights())

# Load data
states = None
actions = None
rewards = None
new_states = None
terminals = None

# Load old data
if os.path.isfile(data_file):
    with h5py.File(data_file, 'r') as f:
        states = f['states'][:]
        actions = f['actions'][:]
        rewards = f['rewards'][:]
        new_states = f['new_states'][:]
        terminals = f['terminals'][:]
        
    os.remove(data_file)

while True:
    t0 = time.time()
    while not os.path.isfile(training_signal):
        # if no training signal for 1 minute, abort
        if time.time() - t0 > 60:
            os.remove(alive_signal)

            # persist data
            model.save_weights(model_file)
            target.save_weights(target_file)
            
            with h5py.File(data_file, 'w') as f:
                f.create_dataset('states', data=states)
                f.create_dataset('actions', data=actions)
                f.create_dataset('rewards', data=rewards)
                f.create_dataset('new_states', data=new_states)
                f.create_dataset('terminals', data=terminals)

            sys.exit()

    # load current parameters
    with open(train_parameters) as json_data:
        config = json.load(json_data)

    epochs = config["epochs"]
    batch_size = config["batch_size"]
    discount_factor = config["discount_factor"]
    target_update_rate = config["target_update_rate"]

    # Load new data
    for new_data_file in os.listdir(new_data_folder):
        if new_data_file.endswith('.h5'):
            # print("Adding data from: %s" % new_data_file)
            with h5py.File(new_data_folder + new_data_file, 'r') as f:
                if states is None:
                    states = f['states'][:]
                    actions = f['actions'][:]
                    rewards = f['rewards'][:]
                    new_states = f['new_states'][:]
                    terminals = f['terminals'][:]
                else:
                    states = np.concatenate([states, f['states'][:]])
                    actions = np.concatenate([actions, f['actions'][:]])
                    rewards = np.concatenate([rewards, f['rewards'][:]])
                    new_states = np.concatenate([new_states, f['new_states'][:]])
                    terminals = np.concatenate([terminals, f['terminals'][:]])
    
            # all data will be saved in data.h5, so no need for old data file
            os.remove(new_data_folder + new_data_file)
    
    if states is not None and states.shape[0]:
        # there is experience to process!
  
        # training
        
        # init experience buffer with indices
        experience_buffer = range(states.shape[0])
        
        for epoch in range(epochs):
    
            # sample a minibatch
            minibatch = np.random.choice(experience_buffer, size=batch_size)
    
            # inputs are the states
            batch_states = np.array([states[i] for i in minibatch])  # bx(sxs)
            batch_actions = np.array([actions[i] for i in minibatch])  # bxa
            batch_rewards = np.array([rewards[i] for i in minibatch])  # bx1
            batch_new_states = np.array([new_states[i] for i in minibatch])  # bx(sxs)
            batch_terminals = np.array([terminals[i] for i in minibatch]).astype('bool')  # bx1

            batch_targets = model.predict_on_batch(batch_states)  # 30, 5
    
            # print("Batch targets Q-Network:\n", batch_targets)
    
            # calculate Q-Values of successor states
            target_q_values = target.predict(batch_new_states)

            # get max_Q values, discount them and set set those values to 0 where state is terminal
            max_target_q_values = np.amax(target_q_values, axis=1)
            # argmax_target_q_values = np.argmax(target_q_values, axis=1)
    
            for k in range(batch_size):
                batch_targets[k, batch_actions[k]] = batch_rewards[k]
                if not batch_terminals[k]:
                    batch_targets[k, batch_actions[k]] += max_target_q_values[k] * discount_factor
                            

            # batch_targets[range(batch_size), batch_actions] = max_target_q_values + batch_rewards
            # print("final targets:\n", batch_targets)
    
            model.train_on_batch(batch_states, batch_targets)
    
            # update target network
            model_weights = model.get_weights()
            target_weights = target.get_weights()
            for i in range(len(model_weights)):
                target_weights[i] = target_update_rate * model_weights[i] + \
                                    (1 - target_update_rate) * target_weights[i]
            target.set_weights(target_weights)

        model.save_weights(model_file)

    os.remove(training_signal)