from Cense.Decider.decider import Decider
import json
import numpy as np
from Cense.Decider.action import Action
import warnings
from keras.models import Model

from Resources.NeuralNetworks.nn import baseline_model as model

#
# Implementation of a Decider making decisions using a Neural Network
#
class DqnDecider(Decider):

    prediction_network = None
    target_network = None

    #
    # Initialize Deep Q-Network
    #
    def __init__(self):
        self.prediction_network = model()
        self.target_network = model()

        self.update_target_network()

    #
    # Returns an action chosen by prediction_network
    #
    def decide(self, state):
        q_values = self.prediction_network(state)
        return np.argmax(q_values)

    #
    # Set weights of Target Network to weights of Prediction Network
    #
    def update_target_network(self):
        self.target_network.set_weights(self.prediction_network.get_weights())