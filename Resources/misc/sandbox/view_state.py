import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

from scipy.misc import imresize

file = "C:\\Users\\Christian\\Thesis\\workspace\\CENSE\\demonstrator_RLAlgorithm\\Resources\\nn-data\\new_data.h5"

with h5py.File(file, 'r') as f:
    states = f['states'][:]

state = states[0]

s = 56

while s >= 1:

    print(s)

    state = imresize(state, (s,s))

    plt.figure()
    plt.imshow(state, cmap='gray')
    plt.draw()
    plt.pause(.001)

    s = s // 2
    s -= 4

plt.show()