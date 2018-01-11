import h5py
import matplotlib.pyplot as plt


file = "path_to_\\data.h5"

with h5py.File(file, 'r') as f:
    states = f['states'][:]

for i in range(len(states)):
    state = states[i]

    state = .5*(state+1)

    plt.figure()
    plt.imshow(state)
    plt.draw()

plt.show()