import Decider.decider
import json
import numpy as np
from Decider.action import Action
import warnings


#
# Implementation of a Decider making decisions based upon a lookup table and an epsilon value
#
class LookupDecider(Decider):

    # Path to the file containing the lookup table
    __lookup_file_path = ""
    # Lookup table
    __lookup = {}
    # Epsilon value determining the probability of a random action
    __epsilon = 0
    # Gamma value determining how much future rewards will be taken into account when calculating new q_values
    __gamma = 0

    #
    # Initiates a lookup table from a given file
    # If no path is given, a standard table will be used
    # Epsilon value determines the probability of a random action
    # Gamma value determines how much future rewards will be taken into account when calculating new q_values
    #
    def __init__(self,  epsilon, gamma, lookup_file='NNs/lookup9.json'):
        # Set path of the lookup table
        self.__lookup_file_path = lookup_file
        # Set gamma value
        self.__gamma = gamma
        self.set_epsilon(epsilon)
        # Open lookup table
        with open(lookup_file, 'r') as fp:
            self.__lookup = json.load(fp)

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
        # Check if we should do a random action
        if np.random.random(1) < self.__epsilon:
            return Action.get_random_action()
        # Check if there is already an entry for this particular state in the table
        elif str(hash_code) in self.__lookup:
            return Action(np.argmax(self.__lookup[str(hash_code)]))
        # Return a random action and create an initial entry in the lookup_table for this state
        else:
            self.__lookup[str(hash_code)] = np.zeros(len([list(Action)]))
            return Action.get_random_action()

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
            self.__lookup[new_hash_code] = np.zeros(len([list(Action)]))
            max_future_q = 0

        # Get q_values of the old state
        hash_code = state_old.hash_code()
        q_values = self.__lookup[str(hash_code)]
        updated_q_values = q_values[:]

        if reward not in [10, 20, -10, -20]:  # non-terminal state
            update = reward + (self.__gamma * max_future_q)
        else:  # terminal state
            update = reward

        # Update the q_values with the formula described in the CENSE paper
        updated_q_values[action.value] = update
        self.__lookup[str(hash_code)] = updated_q_values

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
