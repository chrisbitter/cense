import numpy as np
import hashlib


# Represents a state of the world, a smaller cutout with quadratic shape
class State:
    # Array containing the cutout
    state_array = np.zeros([5, 5, 1], dtype=np.uint8)

    def __init__(self, state_array):
        self.state_array = state_array

    # Returns the hashcode of the state
    def hash_code(self):
        self.state_array.flags.writeable = False
        hash_object = hashlib.sha256(self.state_array.data.tobytes())
        hash_value = hash_object.hexdigest()
        self.state_array.flags.writeable = True
        return hash_value

    # Checks if two states are equal, returns false if their not and true if they are
    def equals(self, state):
        for i in range(self.state_array.shape[0]):
            for j in range(self.state_array.shape[1]):
                if not self.state_array[i][j] == state.__state_array[i][j]:
                    return False
        return True

    # Returns a printable string representation of this State
    def __str__(self):
        print_array = np.empty_like(self.state_array)
        # Rotate array by -90°
        state_size = self.state_array.shape[0]
        for column in range(state_size):
            for row in range(state_size):
                print_array[row, column] = self.state_array[column, state_size - row - 1]
        return str(print_array)
