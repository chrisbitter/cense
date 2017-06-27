import os
from threading import Thread, Lock

import numpy as np
import tensorflow as tf

from Cense.Agent.NeuralNetworkFactory.nnFactory import model_dueling
from Resources.misc.keras_thread.trainer import Trainer


class Agent(object):

    weights_file = "weights.h5"
    lock = Lock()

    def __init__(self):

        self.trainer = Trainer()

        self.model = model_dueling((10, 10), 10)

        X = tf.placeholder("float", [None, 28, 28, 1], name='X')
        Y = tf.placeholder("float", [None, 10], name='Y')

        self.saver = tf.train.Saver()

        with tf.Session(graph=self.graph) as sess:
            tf.train.export_meta_graph('graph')
            # self.saver.save(sess, 'model')


        self.X_data = np.random.rand(1,10,10)
        self.y_data = np.random.randint(0,10,1)



        #self.model.predict_on_batch(self.data)

    def train(self):

        while True:

            if self.trainer.is_done():
                print("Done")
                self.model.load_weights(self.weights_file)

                t = Thread(target=self.trainer.train).start()
                #t.setDaemon(True)
                #t.start()

            else:
                print("Not Done")

            self.model.predict_on_batch(self.data)

            print("end")

    def load(self):
        try:
            with self.lock:
                if os.path.isfile(self.weights_file):

                    self.model.load_weights(self.weights_file)
                    print("load")

                    self.model.predict_on_batch(self.data)
        except:
            raise

    def save(self):
        self.model.save_weights(self.weights_file)

if __name__ == "__main__":

    agent = Agent()

    # for _ in range(2):
    #     Thread(target=agent.load).start()

    # agent.train()
