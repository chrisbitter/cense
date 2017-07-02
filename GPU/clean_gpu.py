import h5py
from keras.models import Model
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.core import Activation, Dense, Flatten, Dropout
from keras.optimizers import RMSprop, Adam
from keras.regularizers import l2
from keras import backend as K
import sys
from keras.models import model_from_json
import numpy as np
import math
import json

from keras.models import Sequential, Model
from keras.layers import Input, Dense, Concatenate, Reshape, Conv2D, MaxPooling2D, Dropout, Flatten, RepeatVector, \
    merge, Activation, Lambda
from keras.layers.merge import Average, Add
import keras.backend as K

from collections import deque

# disables source compilation warnings
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class MissingFileException(Exception):
    def __init__(self, msg):
        self.msg = msg


class Cleaner(object):
    experience_buffer = deque()

    STATE_DIMENSIONS = (40, 40)
    ACTIONS = 5

    root_folder = "/home/useradmin/Dokumente/rm505424/CENSE/Christian/"

    archive_folder = root_folder + "archive/"

    train_parameters = root_folder + "train_params.json"

    model_file = root_folder + "model/model.json"
    weights_file = root_folder + "model/weights.h5"
    target_weights_file = root_folder + "model/target_weights.h5"

    new_data_folder = root_folder + "data/new_data/"
    data_file = root_folder + "data/data.h5"


if __name__ == "__main__":

    try:
        Cleaner.clean()

    except MissingFileException as e:
        print(e.msg)