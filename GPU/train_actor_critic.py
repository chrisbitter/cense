import h5py
from keras import backend as K

import numpy as np
import json

import tensorflow as tf
from collections import deque
import nn_factory

import time

# disables source compilation warnings
import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if len(sys.argv) > 1:
    _ID = sys.argv[1]
else:
    _ID = str(np.random.randint(10000))

STATE_DIMENSIONS = (40, 40)

root_folder = "/home/useradmin/Dokumente/rm505424/CENSE/Christian/training_data/"

train_parameters = root_folder + "train_params.json"

actor_file = root_folder + "model/actor.h5"
critic_file = root_folder + "model/critic.h5"

actor_target_file = root_folder + "model/actor_target.h5"
critic_target_file = root_folder + "model/critic_target.h5"

new_data_folder = root_folder + "data/new_data/"
data_file = root_folder + "data/data.h5"

training_signal = root_folder + "training_signal_" + _ID
alive_signal = root_folder + "alive_signal_" + _ID

with open(alive_signal, 'a'):
    pass

K.set_learning_phase(1)

# create models
actor = nn_factory.actor_network(STATE_DIMENSIONS)
actor_target = nn_factory.actor_network(STATE_DIMENSIONS)
critic = nn_factory.critic_network(STATE_DIMENSIONS)
critic_target = nn_factory.critic_network(STATE_DIMENSIONS)

if os.path.isfile(actor_file):
    actor.load_weights(actor_file)
else:
    raise OSError('actor file missing')

if os.path.isfile(critic_file):
    critic.load_weights(critic_file)

# if use_target:
if os.path.isfile(actor_target_file):
    actor_target.load_weights(actor_target_file)
else:
    actor_target.set_weights(actor.get_weights())

if os.path.isfile(critic_target_file):
    critic_target.load_weights(critic_target_file)
else:
    critic_target.set_weights(critic.get_weights())

action_gradient = tf.placeholder(tf.float32, actor.output_shape)
params_grad = tf.gradients(actor.output, actor.trainable_weights, -action_gradient)
grads = zip(params_grad, actor.trainable_weights)
optimizer = tf.train.AdamOptimizer(0.00001)
optimize = optimizer.apply_gradients(grads)

sess = K.get_session()

# print(critic.inputs)
# print(critic.outputs)

# predicted_q_value = tf.placeholder(tf.float32, [None, 1])
gradients = K.gradients(critic.outputs, critic.inputs[1])[0]  # gradient tensors
get_gradients = K.function(inputs=critic.inputs, outputs=[gradients])

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

use_target = False

while True:
    t0 = time.time()
    while not os.path.isfile(training_signal):
        # if no training signal for 1 minute, abort
        if time.time() - t0 > 60:
            os.remove(alive_signal)

            # persist data
            actor.save_weights(actor_file)
            critic.save_weights(critic_file)
            actor_target.save_weights(actor_target_file)
            critic_target.save_weights(critic_target_file)

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
        experience_buffer = deque(range(states.shape[0]))

        for epoch in range(epochs):

            # sample a minibatch
            minibatch = np.random.choice(experience_buffer, size=batch_size)

            # inputs are the states
            batch_states = np.array([states[i] for i in minibatch])  # bx(sxs)
            batch_actions = np.array([actions[i] for i in minibatch])  # bxa
            batch_rewards = np.array([rewards[i] for i in minibatch])  # bx1
            batch_new_states = np.array([new_states[i] for i in minibatch])  # bx(sxs)
            batch_terminals = np.array([terminals[i] for i in minibatch]).astype('bool')  # bx1

            target_q_values = critic_target.predict_on_batch(
                [batch_new_states, actor_target.predict(batch_new_states)])  # bx1

            y = batch_rewards.copy()  # bx1

            for k in range(batch_size):
                if not batch_terminals[k]:
                    y[k] += discount_factor * target_q_values[k]

            # print(zip(batch_rewards, y, critic.predict_on_batch([batch_states, batch_actions])))

            # evals_before = critic.predict([batch_states, actions_for_gradients])

            # train critic with actual action-state value
            critic.train_on_batch([batch_states, batch_actions], y)
            # get action which would have been chosen by actor
            actions_for_gradients = actor.predict(batch_states)  # bxa

            # get action gradients w.r.t. critic output
            action_grads = get_gradients([batch_states, actions_for_gradients])[0]  # bxa
            # action_grads /= batch_size
            # action_targets = actions_for_gradients + action_grads


            # better_actions = actions_for_gradients + action_grads

            # actor.train_on_batch(batch_states, better_actions)

            # use action gradients to improve action chosen by actor
            sess.run(optimize, feed_dict={
                actor.inputs[0]: batch_states,
                action_gradient: action_grads
            })

            # actions_after = actor.predict(batch_states) # bxa

            # evals_after = critic.predict([batch_states, actor.predict(batch_states)])

            # print(evals_before)
            # print(evals_after)

            # print(evals_after - evals_before)

            # update actor target
            if use_target:
                actor_weights = actor.get_weights()
                actor_target_weights = actor_target.get_weights()
                for i in range(len(actor_weights)):
                    actor_target_weights[i] = target_update_rate * actor_weights[i] + \
                                              (1 - target_update_rate) * actor_target_weights[i]
                actor_target.set_weights(actor_target_weights)

                # update critic target
                critic_weights = critic.get_weights()
                critic_target_weights = critic_target.get_weights()
                for i in range(len(critic_weights)):
                    critic_target_weights[i] = target_update_rate * critic_weights[i] + \
                                               (1 - target_update_rate) * critic_target_weights[i]
                critic_target.set_weights(critic_target_weights)

        actor.save_weights(actor_file)

    # delete training_signal to signal end of training
    os.remove(training_signal)
