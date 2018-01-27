'''Visualization of the filters of VGG16, via gradient ascent in input space.
This script can run on CPU in a few minutes (with the TensorFlow backend).
Results example: http://i.imgur.com/4nj4KjN.jpg
'''
from __future__ import print_function

from scipy.misc import imsave
import numpy as np
import time
from Agent.NeuralNetworkFactory import model_dueling as Model
from keras import backend as K


filename =

with h5py.File(os.path.join(new_data_folder, new_data_file), 'r') as f:
    if self.states is None:
        self.states = f['states'][:]
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