import h5py
import keras.models
from keras import backend as K

import numpy as np
import json

import tensorflow as tf

from collections import deque

# disables source compilation warnings
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

root_folder = "/home/useradmin/Dokumente/rm505424/CENSE/Christian/training_data/"

train_parameters = root_folder + "train_params.json"

actor_file = root_folder + "model/actor.h5"
critic_file = root_folder + "model/critic.h5"

actor_target_file = root_folder + "model/actor_target.h5"
critic_target_file = root_folder + "model/critic_target.h5"

new_data_folder = root_folder + "data/new_data/"
data_file = root_folder + "data/data.h5"

K.set_learning_phase(1)
sess = tf.Session()

# create models
with open(train_parameters) as json_data:
    config = json.load(json_data)

use_target = config["use_target"]
epochs = config["epochs"]
batch_size = config["batch_size"]
discount_factor = config["discount_factor"]
target_update_rate = config["target_update_rate"]

if not os.path.isfile(actor_file):
    raise OSError('actor file missing')

actor = keras.models.load_model(actor_file)

print(actor.output_shape)
action_gradient = tf.placeholder(tf.float32, actor.output_shape)
params_grad = tf.gradients(actor.output, actor.trainable_weights, -action_gradient)
grads = zip(params_grad, actor.trainable_weights)

optimize = tf.train.AdamOptimizer().apply_gradients(grads)

if os.path.isfile(actor_target_file):
    actor_target = keras.models.load_model(actor_target_file)
else:
    actor_target = keras.models.load_model(actor_file)

actor_weights = actor.get_weights()
actor_target_weights = actor_target.get_weights()
equal = [actor_target_weights[i] == actor_weights[i] for i in range(len(actor_weights))]
print(np.shape(equal))
print(np.sum(equal), len(equal))

if not os.path.isfile(critic_file):
    import nn_factory

    critic = nn_factory.critic_network((40, 40))
    critic.save(critic_file)
else:
    critic = keras.models.load_model(critic_file)

predicted_q_value = tf.placeholder(tf.float32, [None, 1])
critic_action_grads = tf.gradients(critic.outputs, critic.inputs[1])  # GRADIENTS for policy update

# gradients = K.gradients(critic.outputs, critic.inputs[1])[0]  # gradient tensors
# get_gradients = K.function(inputs=critic.inputs, outputs=[gradients])

if os.path.isfile(critic_target_file):
    critic_target = keras.models.load_model(critic_target_file)
else:
    critic_target = keras.models.load_model(critic_file)

sess.run(tf.global_variables_initializer())

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

if states is None or not states.size:
    raise OSError("No data to process")

# Save experience data to data.h5 file
with h5py.File(data_file, 'w') as f:
    f.create_dataset('states', data=states)
    f.create_dataset('actions', data=actions)
    f.create_dataset('rewards', data=rewards)
    f.create_dataset('new_states', data=new_states)
    f.create_dataset('terminals', data=terminals)

# training

# init experience buffer
# draw index for sample to avoid copying large amounts of data
experience_buffer = deque(range(states.shape[0]))

q_next_state_expected = critic_target.predict_on_batch([new_states, actor_target.predict(new_states)])

q_expectations = rewards.copy()

for k in range(states.shape[0]):
    if not terminals[k]:
        q_expectations[k] += discount_factor * q_next_state_expected[k]

q_predictions = critic.predict_on_batch([states, actions])

surprise = np.absolute(q_expectations - q_predictions.reshape(q_expectations.shape))

ranks = 1 / (states.shape[0] - np.argsort(surprise))

for epoch in range(epochs):

    # sample a minibatch
    minibatch = np.random.choice(experience_buffer, size=batch_size, p=ranks)

    # inputs are the states
    batch_states = np.array([states[i] for i in minibatch])  # 30, 20, 20
    batch_actions = np.array([actions[i] for i in minibatch])
    batch_rewards = np.array([rewards[i] for i in minibatch])  # 30, 1
    batch_new_states = np.array([new_states[i] for i in minibatch])  # 30, 20, 20
    batch_terminals = np.array([terminals[i] for i in minibatch]).astype('bool')  # 30, 1

    target_q_values = critic_target.predict_on_batch([batch_new_states, actor_target.predict(batch_new_states)])

    y = batch_rewards.copy()

    for k in range(batch_size):
        if not batch_terminals[k]:
            y[k] += discount_factor * target_q_values[k]

    # train critic with actual action-state value
    loss = critic.train_on_batch([batch_states, batch_actions], y)
    # get action which would have been chosen by actor
    actions_for_gradients = actor.predict(batch_states)

    # get action gradients w.r.t. critic output
    action_grads = sess.run(critic_action_grads, feed_dict={
        critic.inputs[0]: batch_states,
        critic.inputs[1]: actions_for_gradients
    })[0]

    # use action gradients to improve action chosen by actor
    sess.run(optimize, feed_dict={
        actor.inputs[0]: batch_states,
        action_gradient: action_grads
    })

    # update actor target
    actor_weights = actor.get_weights()
    actor_target_weights = actor_target.get_weights()
    for i in range(len(actor_weights)):
        actor_target_weights[i] = target_update_rate * actor_weights[i] + (1 - target_update_rate) * \
                                                                          actor_target_weights[i]
    actor_target.set_weights(actor_target_weights)

    # update critic target
    critic_weights = critic.get_weights()
    critic_target_weights = critic_target.get_weights()
    for i in range(len(critic_weights)):
        critic_target_weights[i] = target_update_rate * critic_weights[i] + (1 - target_update_rate) * \
                                                                            critic_target_weights[i]
    critic_target.set_weights(critic_target_weights)

actor.save(actor_file)
actor_target.save(actor_target_file)
critic.save(critic_file)
critic_target.save(critic_target_file)
