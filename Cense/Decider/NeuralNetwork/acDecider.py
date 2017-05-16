import numpy as np
from keras.models import load_model

from Cense.Decider.action import Action
from Cense.Decider.decider import Decider
from Cense.NeuralNetworkFactory.nnFactory import model_ac as model


#
# Implementation of a Decider making decisions using a Neural Network
#
class AcDecider(Decider):

    __neural_network = None

    #
    # Initialize Neural Network
    #
    def __init__(self):
        self.__neural_network = model()

    #
    # Returns an action chosen by prediction_network
    #
    def decide(self, state):
        q_values = None
        if np.random.random(1) < self.__epsilon:
            return Action.get_random_action()
        else:
            self.prediction_network(state)
            return Action(np.argmax(q_values))

    def load_network(self, filepath):
        self.__neural_network = load_model(filepath)

    def save_network(self, filepath):
        self.__neural_network = self.__neural_network.save(filepath)

if __name__ == "__main__":
    decider = NeuralNetworkDecider(.1)
