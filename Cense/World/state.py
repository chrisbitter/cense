import numpy as np


# Represents a state of the world, a smaller cutout with quadratic shape
class State:

    # Array containing the cutout
    __state_array = np.zeros([5, 5, 1], dtype=np.uint8)

    def __init__(self, state_array):
        self.__state_array = state_array

    # Returns the hashcode of the cutout
    def hash_code(self):
        return hash(self.__state_array)

    # Checks if two states are equal, returns false if their not and true if they are
    def equals(self, state):
        for i in range(self.__state_array.shape[0]):
            for j in range(self.__state_array.shape[1]):
                if not self.__state_array[i][j] == state.__state_array[i][j]:
                    return False
        return True
