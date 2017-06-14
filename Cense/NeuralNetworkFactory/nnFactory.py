from keras.models import Sequential, Model
from keras.layers import Input, Dense, Concatenate, Reshape, Conv2D, MaxPooling2D, Dropout, Flatten, RepeatVector, merge, Activation, Lambda
from keras.layers.merge import Average, Add
import keras.backend as K
from tensorflow import reduce_mean

import tensorflow as tf

import numpy as np

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def model_simple_conv(input_shape, output_dim):
    model = Sequential()

    model.add(Reshape(input_shape + (1,), input_shape=input_shape))
    model.add(Conv2D(30, kernel_size=(5, 5), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(15, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dense(50, activation="relu"))
    model.add(Dense(output_dim, activation="tanh"))

    model.compile(loss='mse',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model

# v1
def model_dueling(input_side_length, num_outputs):

    graph = tf.Graph()

    with graph.as_default():

        #input assumed to be quadratic
        x = tf.placeholder(tf.float32, shape=[None, input_side_length, input_side_length])
        y_ = tf.placeholder(tf.float32, shape=[None, 5])

        x_image = tf.reshape(x, [-1, input_side_length, input_side_length, 1])

        keep_prob = tf.placeholder(tf.float32)

        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

        h_pool2_flat = tf.reshape(h_pool2, [-1, 10 * 10 * 64])

        # advantage
        W_fc_adv_1 = weight_variable([10 * 10 * 64, 250])
        b_fc_adv_1 = bias_variable([250])
        h_fc_adv_1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc_adv_1) + b_fc_adv_1)
        h_fc_adv_1_drop = tf.nn.dropout(h_fc_adv_1, keep_prob)

        W_fc_adv_2 = weight_variable([250, num_outputs])
        b_fc_adv_2 = bias_variable([num_outputs])
        h_fc_adv_2 = tf.nn.tanh(tf.matmul(h_fc_adv_1_drop, W_fc_adv_2) + b_fc_adv_2)
        h_fc_adv_2_drop = tf.nn.dropout(h_fc_adv_2, keep_prob)

        # value
        W_fc_val_1 = weight_variable([10 * 10 * 64, 250])
        b_fc_val_1 = bias_variable([250])
        h_fc_val_1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc_val_1) + b_fc_val_1)
        h_fc_val_1_drop = tf.nn.dropout(h_fc_val_1, keep_prob)

        W_fc_val_2 = weight_variable([250, 1])
        b_fc_val_2 = bias_variable([1])
        h_fc_val_2 = tf.nn.tanh(tf.matmul(h_fc_val_1_drop, W_fc_val_2) + b_fc_val_2)
        h_fc_val_2_drop = tf.nn.dropout(h_fc_val_2, keep_prob)


        y_ = tf.sub(tf.add(h_fc_adv_2_drop, h_fc_val_2_drop), reduce_mean(h_fc_adv_2_drop))

    return graph

def model_dueling_keras(input_shape, output_dim):
    # Common Layers
    input_layer = Input(shape=input_shape)
    common_layer = Reshape(input_shape + (1,))(input_layer)
    common_layer = Conv2D(30, (5, 5), activation="relu")(common_layer)
    # common_layer = MaxPooling2D(pool_size=(2, 2))(common_layer)
    common_layer = Conv2D(15, (5, 5), activation="relu")(common_layer)
    # common_layer = MaxPooling2D(pool_size=(2, 2))(common_layer)
    common_layer = Flatten()(common_layer)

    #adv_layer = Dropout(0.2)(common_layer)
    adv_layer = Dense(100, activation="relu")(common_layer)
    adv_layer = Dense(50, activation="relu")(adv_layer)
    adv_layer = Dense(output_dim, activation="tanh")(adv_layer)

    #val_layer = Dropout(0.2)(common_layer)
    val_layer = Dense(100, activation="relu")(common_layer)
    val_layer = Dense(50, activation="relu")(val_layer)
    val_layer = Dense(1, activation="linear")(val_layer)
    val_layer = RepeatVector(output_dim)(val_layer)
    val_layer = Flatten()(val_layer)
    # q = v + a - mean(a, reduction_indices=1, keep_dims=True)
    #q_layer = val_layer + adv_layer - reduce_mean(adv_layer, keep_dims=True)

    q_layer = merge(inputs=[adv_layer, val_layer], mode=lambda x: x[1] + x[0] - K.mean(x[0], keepdims=True),
                        output_shape=lambda x: x[0])
    #q_layer = Activation(activation="tanh")(q_layer)

    model = Model(inputs=[input_layer], outputs=[q_layer])

    model.compile(loss='mse',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model

def model_ac(input_shape, output_dim):
    # Common Layers
    input_layer = Input(shape=input_shape)
    common_layer = Conv2D(30, (5, 5), input_shape=input_shape, activation="relu")(input_layer)
    common_layer = MaxPooling2D(pool_size=(2, 2))(common_layer)
    common_layer = Conv2D(15, (3, 3), activation="relu")(common_layer)
    common_layer = MaxPooling2D(pool_size=(2, 2))(common_layer)
    common_layer = Flatten()(common_layer)

    pol_layer = Dropout(0.2)(common_layer)
    pol_layer = Dense(128, activation="relu")(pol_layer)
    pol_layer = Dense(50, activation="relu")(pol_layer)
    pol_layer = Dense(output_dim, activation="tanh")(pol_layer)

    val_layer = Dropout(0.2)(common_layer)
    val_layer = Dense(128, activation="relu")(val_layer)
    val_layer = Dense(50, activation="relu")(val_layer)
    val_layer = Dense(1, activation="linear")(val_layer)
    val_layer = RepeatVector(output_dim)(val_layer)
    val_layer = Flatten()(val_layer)
    # q = v + a - mean(a, reduction_indices=1, keep_dims=True)
    merge_layer = merge(inputs=[pol_layer, val_layer], mode=lambda x: x[1] + x[0] - K.mean(x[0], keepdims=True),
                        output_shape=lambda x: x[0])
    merge_layer = Activation(activation="softmax")(merge_layer)

    model = Model(inputs=[input_layer], outputs=[merge_layer])

    model.compile(loss='mse',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


if __name__ == "__main__":
    model = model_dueling((28, 28, 1), 6)

    X = np.random.random((10, 28, 28, 1)).astype('float32')

    # y = np.random.random((100, 6, 1))
    # print(X.shape)

    # X = X.reshape(len(X), 1, 28, 28).astype('float32')

    for i in range(5):
        X = np.random.random((100, 28, 28, 1)).astype('float32')
        y = np.random.randint(2, (100, 6))
        model.fit(X, y)

    # print(X.shape)
    # print(y.shape)
    # print(X)
    # print(y)


    for i in range(5):
        X = np.random.random((100, 28, 28, 1)).astype('float32')
        y = model.predict(X)
        print(y.argmax(axis=1))


def baseline_model():
    model = Sequential()
    model.add(Dense(units=64, input_shape=100))
    model.add(Activation('relu'))
    model.add(Dense(units=10))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])

    return model
