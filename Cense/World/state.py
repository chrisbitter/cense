import numpy as np


# Represents a state of the world, a smaller cutout with quadratic shape
class State:
    # Array containing the cutout
    __state_array = np.zeros([5, 5, 1], dtype=np.uint8)

    def __init__(self, state_array):
        self.__state_array = state_array

    # Returns the hashcode of the state
    def hash_code(self):
        return hash(self.__state_array)

    # Checks if two states are equal, returns false if their not and true if they are
    def equals(self, state):
        for i in range(self.__state_array.shape[0]):
            for j in range(self.__state_array.shape[1]):
                if not self.__state_array[i][j] == state.__state_array[i][j]:
                    return False
        return True

    # Returns a printable string representation of this State
    def __str__(self):
        print_array = np.empty_like(self.__state_array)
        # Rotate array by -90°
        state_size = self.__state_array.shape[0]
        for column in range(state_size):
            for row in range(state_size):
                print_array[row, column] = self.__state_array[column, state_size - row - 1]
        return str(print_array)
