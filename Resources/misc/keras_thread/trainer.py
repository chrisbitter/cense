import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
import os

#define baseline model
def baseline_model(num_pixels, num_classes):
    #create model
    model = Sequential()
    model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu', name='dense1'))
    model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax', name='dense2'))
    #compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

class Trainer(object):

    done = True
    weights_file = "weights.h5"

    def __init__(self):
        # fix random number for reproducability
        seed = 7
        numpy.random.seed(seed)

        # load data
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

        # flatten 28*28 images to 784 vector
        self.num_pixels = X_train.shape[1] * X_train.shape[2]
        X_train = X_train.reshape(X_train.shape[0], self.num_pixels).astype('float32')
        X_test = X_test.reshape(X_test.shape[0], self.num_pixels).astype('float32')

        # normalize inputs from 0-255 to 0-1
        self.X_train = X_train / 255
        self.X_test = X_test / 255

        # one hot encode outputs
        self.y_train = np_utils.to_categorical(y_train)
        self.y_test = np_utils.to_categorical(y_test)
        self.num_classes = self.y_test.shape[1]
        # build model
        #self.model = baseline_model(self.num_pixels, self.num_classes)


    def train(self):

        print("train")

        #self.model.load_weights(self.weights_file)
        self.done = False
        # fit model
        self.model.fit(self.X_train, self.y_train, validation_data=(self.X_test, self.y_test), epochs=10, batch_size=200, verbose=2)
        # Final evaluation
        scores = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        print("Baseline Error: %.2f%%" % (100 - scores[1] * 100))

        self.done = True

    def is_done(self):
        return self.done

if __name__ == "__main__":

    trainer = Trainer()
    trainer.train()
