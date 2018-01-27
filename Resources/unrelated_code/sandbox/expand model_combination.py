import numpy as np

from NeuralNetwork import nnFactory as Factory

model = Factory.model_dueling((50, 50), 5)

state = np.random.random((50,50)) - .5

model_expanded = Factory.model_dueling((50, 50), 15)

identical_layers = {'conv_1', 'conv_2', 'adv_1', 'adv_2', 'val_1', 'val_2', 'val_3', 'q_layer'}

for layer in identical_layers:
    model_expanded.get_layer(layer).set_weights(model.get_layer(layer).get_weights())

adv_layer = model.get_layer(name='adv_3')
adv_weights = adv_layer.get_weights()

adv_W = adv_weights[0]
adv_b = adv_weights[1]

print(np.shape(adv_b))

adv_expanded_weights = np.array([np.concatenate((adv_W, adv_W, adv_W), axis=1), np.concatenate((adv_b, adv_b, adv_b))])

adv_layer_expanded = model_expanded.get_layer(name='adv_3')
adv_layer_expanded.set_weights(adv_expanded_weights)

print(model.predict(np.expand_dims(state, axis=0)))
print(model_expanded.predict(np.expand_dims(state, axis=0)))