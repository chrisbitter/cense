import nnFactory as nnF
from keras import backend as K
import numpy as np
import cv2 as cv
# from matplotlib import pyplot as plt

'''
This imports the nnFactory script that builds the actor model just like it is built before training, this then
generates an input from a sample image to be able to calculate the activation maps. After that is done it then squeezes the
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

class Visualizer():
    def load_models(self):
        actor_model = nnF.actor_network(input_shape)  # input_size = (40, 40, 3)
        critic_model = nnF.critic_network(input_shape)  # input_size = (40, 40, 3)
        return actor_model, critic_model


    def prepare_theano_functions(self):
        actor_model, critic_model = self.load_models()

        actor_model.load_weights('actor.h5')
        actor_inp = actor_model.input                                            # input placeholder
        actor_outputs = [layer.output for layer in actor_model.layers]           # all layer outputs
        actor_functor = K.function([actor_inp] + [K.learning_phase()], actor_outputs)  # evaluation function

        critic_inp = critic_model.input                                            # input placeholder
        critic_outputs = [layer.output for layer in critic_model.layers]           # all layer outputs
        critic_functor = K.function([critic_inp] + [K.learning_phase()], critic_outputs)  # evaluation function
        return actor_functor, critic_functor


    def testing_functor(self):
        actor_functor, critic_functor = self.prepare_theano_functions()
        im = cv.imread('cense_input_raw.png', cv.IMREAD_COLOR)
        resize_im = cv.resize(im,(40,40))
        # test = np.random.random(input_shape)[np.newaxis, ...]  # This creates Random Input
        test = resize_im[np.newaxis, ...]  # This loads a sample picture as input
        print(type(test[0][0][0][0]))
        actor_layer_outs = actor_functor([test, 1.])
        # critic_layer_outs = critic_functor([test, 1.])
        critic_layer_outs = None
        return actor_layer_outs, critic_layer_outs


    def generate_element_weights_vektor(self):
        actor_layer_outs, critic_layer_outs = self.testing_functor()
        element_weights_vektor = []
        layer_counter = 0
        for alayer in actor_layer_outs:
            if not (layer_counter in [6, 7, 9, 11, 13]):
                print(alayer[0].shape)
                try:
                    i_max = len(alayer[0])
                    j_max = len(alayer[0][0])
                    k_max = len(alayer[0][0][0])
                    for k in range(k_max):
                        for i in range(i_max):
                            for j in range(j_max):
                                element_weights_vektor.append(alayer[0][i][j][k].item())
                except:
                    j_max = len(alayer[0])
                    for j in range(j_max):
                        element_weights_vektor.append(alayer[0][j].item())
            layer_counter += 1
        #
        # for clayer in critic_layer_outs:
        #     try:
        #         i_max = len(clayer[0])
        #         j_max = len(clayer[0][0])
        #         k_max = len(clayer[0][0][0])
        #         for k in range(k_max):
        #             for j in range(j_max):
        #                 for i in range(i_max):
        #                     clayer[0][i][j][k] = np.asscalar(clayer[0][i][j][k])
        #                     element_weights_vektor.append(clayer[0][i][j][k])
        #     except:
        #         j_max = len(clayer[0])
        #         for j in range(j_max):
        #             clayer[0][j] = np.asscalar(clayer[0][j])
        #             element_weights_vektor.append(clayer[0][j])
        #             # print(layer[0][j])
        return element_weights_vektor


if __name__ == "__main__":

    visualizer = Visualizer()

    element_weights_vektor = visualizer.generate_element_weights_vektor()
    # soll = 40 * 40 * 3 + 36 * 36 * 30 + 18 * 18 * 30 + 14 * 14 * 15 + 7 * 7 * 15 + 5 * 5 * 10 + 250 + 250 + 400 + 400 + 200 + 200 + 100 + 100 + 1 + 1 + 1 + 3
    image = []
    for i in range(40):
        image.append(element_weights_vektor[i*40:(i+1)*40])
    # plt.imshow(image)
    # plt.show()

    soll = 40*40*3+36*36*30+18*18*30+14*14*15+7*7*15+5*5*10+400+200+100+1+1+1+3
    print(soll)
    print(len(element_weights_vektor))
    print(element_weights_vektor)

