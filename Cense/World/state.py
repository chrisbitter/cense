import numpy as np


class State:

    __state_array = np.zeros([5, 5, 1], dtype=np.uint8)

    # World encoding
    wire = 1
    rotor_top = 2
    rotor_bot = 4
    goal = 8

    def __init__(self):
        print("initiated")
