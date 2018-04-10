import h5py
import matplotlib.pyplot as plt
import os

path_to_data = os.path.abspath(os.path.join(os.getcwd(), "../nn-data/new_data.h5"))

with h5py.File(path_to_data, 'r') as f:
    states = f['states'][:]

print(len(states))

for i in range(len(states)):
    state = states[i]

    state = (state + 1) / 2

    plt.figure()
    plt.imshow(state, vmin=0, vmax=1)

    plt.show()