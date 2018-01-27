import h5py
import matplotlib.pyplot as plt


file = "C:/Users/Christian/Thesis/workspace/CENSE/demonstrator_RLAlgorithm/Resources/nn-data/data.h5"

with h5py.File(file, 'r') as f:
    states = f['states'][:]

for i in range(len(states)):
    state = states[i]

    plt.figure()
    plt.imshow(state, vmin=0, vmax=1)

    plt.show()