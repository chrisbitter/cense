import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from Cense.Agent.NeuralNetworkFactory import nnFactory as Factory
import time

class Visualizer(object):
    def __init__(self):

        self.model = Factory.model_acceleration_q((50,50), (3,), 27)
        self.model.load_weights('model/weights.h5')

        new_data_folder = os.path.dirname(os.path.abspath(__file__))

        self.states = None
        self.actions = None
        self.rewards = None
        self.suc_states = None
        self.terminals = None

        for new_data_file in os.listdir(new_data_folder):
            if new_data_file.endswith('.h5'):
                # print("Adding data from: %s" % new_data_file)
                with h5py.File(os.path.join(new_data_folder, new_data_file), 'r') as f:
                    if self.states is None:
                        self.states = f['states'][:]
                        self.velocities = f['velocities'][:]
                        self.actions = f['actions'][:]
                        self.rewards = f['rewards'][:]
                        self.suc_states = f['suc_states'][:]
                        self.terminals = f['terminals'][:]
                    else:
                        self.states = np.concatenate([self.states, f['states'][:]])
                        self.actions = np.concatenate([self.actions, f['actions'][:]])
                        self.rewards = np.concatenate([self.rewards, f['rewards'][:]])
                        self.suc_states = np.concatenate([self.suc_states, f['suc_states'][:]])
                        self.terminals = np.concatenate([self.terminals, f['terminals'][:]])

    def show(self):
        print("show")
        fig = plt.figure(0)

        rows = 1
        cols = 1

        ii = 1

        for index in range(len(self.states)//(rows*cols)):
            for row in range(rows):
                for col in range(cols):
                    plt.subplot(rows, cols, col + row*cols + 1)
                    plt.axis('off')
                    plt.imshow(self.states[index*rows*cols + col + row*cols], cmap='gray').norm.vmax = 1

                    #plt.imsave(, arr=self.states[index*rows*cols + col + row*cols])

                    #ii += 1

                    # plt.subplot(rows, 2*cols, 2 * col + row * cols + 2)
                    # plt.axis('off')
                    # if not self.terminals[index]:
                    #     plt.imshow(self.suc_states[index], cmap='gray').norm.vmax = 1
                    # else:
                    #     plt.imshow(np.ones(self.suc_states[index*rows*cols + col + row*cols].shape), cmap='gray').norm.vmax = 1

                    q_vals = self.model.predict([np.expand_dims(self.states[index*rows*cols + col + row*cols], axis=0), np.expand_dims(self.velocities[index*rows*cols + col + row*cols], axis=0)])

                    print(self.terminals[index*rows*cols + col + row*cols])
                    print(self.actions[index*rows*cols + col + row*cols])
                    print(q_vals)

            #plt.tight_layout()
            plt.draw()
            plt.pause(.001)
            input()
            #



if __name__ == "__main__":
    vis = Visualizer()
    vis.show()