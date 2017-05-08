from Cense.Decider.decider import Decider
import json
import numpy as np
from Cense.Decider.action import Action
import warnings
from keras.models import Sequential

#
# Implementation of a Decider making decisions using a Neural Network
#
class DqnDecider(Decider):

    prediction_network = None
    target_network = None

    #
    #
    #
    def __init__(self):
        self.prediction_network = create_model()
        self.target_network = create_model()

        target_network

    def create_network(self):
        model = Sequential()

    #
    # Persists the lookup table into a json file.
    # If no path is specified the table will be written into the file it was read from
    #
    def persist_lookup_table(self, lookup_file=None):
        # Determine path
        file_path = self.__lookup_file_path
        if lookup_file is not None:
            file_path = lookup_file

        # Save at determined path
        with open(file_path, 'w') as fp:
            json.dump(self.__lookup, fp)

    #
    # Returns an action based on the epsilon value and the lookup_table
    #
    def decide(self, state):
        hash_code = state.hash_code()
        # Check if state has an entry, if not create one and return random action
        if not str(hash_code) in self.__lookup:
            print("creating new entry for key: " + str(hash_code))
            self.__lookup[str(hash_code)] = np.zeros(len([(action.value, action.name) for action in Action])).tolist()
            return Action.get_random_action()
        # Check if we should do a random action
        elif np.random.random(1) < self.__epsilon:
            return Action.get_random_action()
        # if there is already an entry for this particular state in the table return the corresponding action
        else:
            return Action(np.argmax(self.__lookup[str(hash_code)]))
        # Return a random action and create an initial entry in the lookup_table for this state

    #
    # Updates the Q-value in the lookup table for a specific state and action
    #
    def update_q(self, state_old, state_new, action, reward):
        # Find maximum future q_value
        new_hash_code = str(state_new.hash_code())

        # Check for existing entries
        if new_hash_code in self.__lookup:
            max_future_q = np.max(self.__lookup[new_hash_code])
        else:
            # Create initial entry
            self.__lookup[new_hash_code] = np.zeros(len([(action.value, action.name) for action in Action])).tolist()
            max_future_q = 0

        # Get q_values of the old state
        hash_code = state_old.hash_code()
        q_values = np.array(self.__lookup[str(hash_code)])
        updated_q_values = q_values[:]

        if reward not in [10, 20, -10, -20]:  # non-terminal state
            update = reward + (self.__gamma * max_future_q)
        else:  # terminal state
            update = reward

        # Update the q_values with the formula described in the CENSE paper
        updated_q_values[action.value] = update
        self.__lookup[str(hash_code)] = updated_q_values.tolist()

    #
    # Sets new epsilon value
    # Should be used after finishing an epoch
    #
    def set_epsilon(self, epsilon):
        # Check if epsilon value is right
        if 1 >= epsilon >= 0:
            self.__epsilon = epsilon
        # If epsilon value is wrong, assume no exploration should be done
        else:
            warnings.warn("Epsilon value should be between 0 and 1, will continue with epsilon = 0.9")
            self.__epsilon = 0.9
