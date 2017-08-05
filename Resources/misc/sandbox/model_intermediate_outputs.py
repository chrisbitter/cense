from Cense.Agent.NeuralNetworkFactory.nnFactory import actor_network
import os
import h5py
import numpy as np
import time
from keras.models import Model
import keras.backend as K

file = "C:\\Users\\Christian\\Thesis\\workspace\\CENSE\\demonstrator_RLAlgorithm\\Resources\\nn-data\\new_data.h5"
weights = "C:\\Users\\Christian\\Thesis\\workspace\\CENSE\\demonstrator_RLAlgorithm\\Resources\\nn-data\\actor.h5"

actor = actor_network((40,40))

print(actor.summary())

actor.load_weights(weights)

layers = ["conv_1", "conv_2", "mlp_1",  "mlp_2"]
layers = ["conv_3"]

models = {}

for layer in layers:
    models[layer] = K.function([actor.input, K.learning_phase()], [actor.get_layer(layer).output])
    #model = Model(inputs=[actor.input], outputs=[actor.get_layer(layer)])
    #models[layer] = model
    print("appending %s" % layer)


# with h5py.File(file, 'r') as f:
#
#     states = f['states'][:]
#     actions = f['actions'][:]
#     rewards = f['rewards'][:]
#     suc_states = f['new_states'][:]
#     terminals = f['terminals'][:]

# for i in range(states.shape[0]):
#     print(i)
#print(intermediate.predict(np.expand_dims(states[0], axis=0)))
for i in range(1):
    print("run %d" % i)
    state = np.random.random((1,40,40)) * 2 - 1
    for layer in layers:
        print(layer)
        print("out")
        out = models[layer]([state,0])
        print("min: %f\nmax: %f" % (np.min(out), np.max(out)))
        print("weights")
        weights = actor.get_layer(layer).get_weights()[0]
        print("min: %f\nmax: %f" % (np.amin(weights), np.amax(weights)))
    print("output")
    print(actor.predict(state))

print("random outputs")
for i in range(1):
    #state = np.random.random((1,40,40)) * 2 - 1
    state = np.zeros((1,40,40))
    state[:,:,20:25] = np.ones((1,40,5))
    print(actor.predict(state))

    state = np.zeros((1, 40, 40))
    state[:, 20:25, :] = np.ones((1, 5, 40))
    print(actor.predict(state))

#print(actor_weights[1])

state = np.ones((1,40,40))
for layer in layers:
    print(layer)
    print("out")
    out = models[layer]([state,0])
    print("min: %f\nmax: %f" % (np.min(out), np.max(out)))
    print("weights")
    weights = actor.get_layer(layer).get_weights()[0]
    print("min: %f\nmax: %f" % (np.amin(weights), np.amax(weights)))

state = -np.ones((1,40,40))
for layer in layers:
    print(layer)
    print("out")
    out = models[layer]([state,0])
    print("min: %f\nmax: %f" % (np.min(out), np.max(out)))
    print("weights")
    weights = actor.get_layer(layer).get_weights()[0]
    print("min: %f\nmax: %f" % (np.amin(weights), np.amax(weights)))