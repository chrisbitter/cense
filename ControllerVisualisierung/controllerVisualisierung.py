from keras import backend as K
import numpy as np
import threading
import socket
import tensorflow as tf
import sys
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


class VisSocket:
    def __init__(self):
        self.TCP_IP = '137.226.189.157'
        self.TCP_PORT = 59595
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def connect(self):
        self.sock.connect((self.TCP_IP, self.TCP_PORT))

    def send_package(self, package):
        # print(len(package))
        package = np.array(package, dtype="float32")
        # print('max: {}  min: {}  median: {}'.format(np.amax(package), np.amin(package), np.median(package)))
        byte_package = package.tobytes()
        self.sock.send(byte_package)
        del byte_package

class Visualizer:
    def __init__(self, tf_graph):
        self.graph = tf_graph
        self.input_shape = (40, 40, 3)
        self.lock = threading.Lock()
        self.element_weights_vector = []
        self.actor_model = None
        self.critic_model = None
        self.state = None
        self.t = threading.Thread(target=self.generate_element_weights_vector, name='vector_generator')

        self.vis_socket = VisSocket()
        self.comm_tries = 10
        self.vis_socket.connect()

    def prepare_theano_functions(self):
        with self.graph.as_default():
            actor_inp = self.actor_model.input                                            # input placeholder
            actor_outputs = [layer.output for layer in self.actor_model.layers]           # all layer outputs
            actor_functor = K.function([actor_inp] + [K.learning_phase()], actor_outputs)  # evaluation function

        # critic_inp = critic_model.input                                            # input placeholder
        # critic_outputs = [layer.output for layer in critic_model.layers]           # all layer outputs
        # critic_functor = K.function([critic_inp] + [K.learning_phase()], critic_outputs)  # evaluation function
        critic_functor = None
        return actor_functor, critic_functor

    def testing_functor(self):
        actor_functor, critic_functor = self.prepare_theano_functions()
        test = self.state[np.newaxis, ...]  # This loads a sample picture as input
        actor_layer_outs = actor_functor([test, 1.])
        # critic_layer_outs = critic_functor([test, 1.])
        critic_layer_outs = None
        return actor_layer_outs, critic_layer_outs

    def send_vector(self):
        try:
            self.vis_socket.send_package(self.element_weights_vector)
        except RuntimeError:
            pass

    def generate_element_weights_vector(self):

        if self.lock.acquire(False):
            try:
                self.element_weights_vector = []

                actor_layer_outs, critic_layer_outs = self.testing_functor()
                layer_counter = 0
                for alayer in actor_layer_outs:
                    if not (layer_counter in [6, 7, 9, 11, 13]):
                        try:
                            i_max = len(alayer[0])
                            j_max = len(alayer[0][0])
                            k_max = len(alayer[0][0][0])
                            for k in range(k_max):
                                for i in range(i_max):
                                    for j in range(j_max):
                                        self.element_weights_vector.append(alayer[0][i][j][k].item())
                        except:
                            j_max = len(alayer[0])
                            for j in range(j_max):
                                self.element_weights_vector.append(alayer[0][j].item())
                    layer_counter += 1

                self.send_vector()
                #print('Active Threads: {}'.format(threading.active_count()))
            except:
                pass

            self.lock.release()

    def visualize(self, model, state, graph):
        if not self.t.is_alive():
            self.actor_model = model
            self.state = state
            self.graph = graph
            del self.t
            self.t = threading.Thread(target=self.generate_element_weights_vector, name='vector_generator')
            self.t.start()
            # print("Active Threads: {}".format(threading.enumerate()))
            # t.join()  # just for testing


if __name__ == "__main__":
    pass
    '''
    graph = tf.get_default_graph()
    visualizer = Visualizer(graph)
    actor_model, critic_model = visualizer.load_models()
    im = cv.imread('cense_input_raw.png', cv.IMREAD_COLOR)
    resize_im = cv.resize(im, (40, 40))
    for i in range(10):
        visualizer.visualize(actor_model, resize_im)

    # soll = 40 * 40 * 3 + 36 * 36 * 30 + 18 * 18 * 30 + 14 * 14 * 15 + 7 * 7 * 15 + 5 * 5 * 10 + 250 + 250 + 400 + 400 + 200 + 200 + 100 + 100 + 1 + 1 + 1 + 3
    image = []
    for i in range(40):
        image.append(visualizer.element_weights_vektor[i*40:(i+1)*40])
    # plt.imshow(image)
    # plt.show()

    soll = 40*40*3+36*36*30+18*18*30+14*14*15+7*7*15+5*5*10+400+200+100+1+1+1+3
    print(soll)
    print(len(visualizer.element_weights_vektor))
    # print(visualizer.element_weights_vektor)

    '''
