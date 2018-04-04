import nnFactory as nnF
from keras import backend as K
import numpy as np

'''
This imports the nnFactory script that builds the actor model just like it is built before training, this then
generates a random input to be able to calculate the activation maps. After that is done it then squeezes the
multidimensional numpy array into a one dimensional data structured vector according to the formatting stated by
VCI:


element weights vector structure:
for vector of weights at index x: element_weights_vector[x] with:
x is index of first element of the first neuron in the current layer.
i: number of elements in current row -> sqrt(number of elements in neuron)
j: number of elements in current collumn -> j = i
k: number of neurons in current layer
weight of next element in row direction: element_weights_vector[x + 1]
weight of next element in collumn direction: element_weights_vector[x + i]
weight of first element in next neuron: element_weights_vector[x + i * j]
weight of first element in next layer: element_weights_vector[x + i * j * k]


To generate example element_weights_vektor the method to import is generate_element_weights_vektor().
'''

input_shape = (40, 40, 3)


def load_models():
    actor_model = nnF.actor_network(input_shape)  # input_size = (40, 40, 3)
    critic_model = nnF.critic_network(input_shape)  # input_size = (40, 40, 3)
    return actor_model, critic_model


def prepare_theano_functions():
    model, _ = load_models()

    inp = model.input                                            # input placeholder
    outputs = [layer.output for layer in model.layers]           # all layer outputs
    functor = K.function([inp] + [K.learning_phase()], outputs)  # evaluation function
    return functor


def testing_functor():
    functor = prepare_theano_functions()
    test = np.random.random(input_shape)[np.newaxis, ...]
    layer_outs = functor([test, 1.])
    return layer_outs


def generate_element_weights_vektor():
    layer_outs = testing_functor()
    element_weights_vektor = []
    for layer in layer_outs:
        try:
            i_max = len(layer[0])
            j_max = len(layer[0][0])
            k_max = len(layer[0][0][0])
            for k in range(k_max):
                for j in range(j_max):
                    for i in range(i_max):
                        layer[0][i][j][k] = np.asscalar(layer[0][i][j][k])
                        element_weights_vektor.append(layer[0][i][j][k])
        except:
            j_max = len(layer[0])
            for j in range(j_max):
                layer[0][j] = np.asscalar(layer[0][j])
                element_weights_vektor.append(layer[0][j])
                # print(layer[0][j])
    return element_weights_vektor


if __name__ == "__main__":
    element_weights_vektor = generate_element_weights_vektor()
    soll = 40 * 40 * 3 + 36 * 36 * 30 + 18 * 18 * 30 + 14 * 14 * 15 + 7 * 7 * 15 + 5 * 5 * 10 + 250 + 250 + 400 + 400 + 200 + 200 + 100 + 100 + 1 + 1 + 1 + 3

    print(len(element_weights_vektor))
    print(soll)
