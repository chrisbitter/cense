from keras.models import Sequential, Model
from keras.layers import Input, Dense, Concatenate, Reshape, Conv2D, MaxPooling2D, Dropout, Flatten, RepeatVector, \
    merge, Activation, Lambda, Multiply
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


class QNetwork():
    def __init__(self, input_dimensions, num_actions):
        # input assumed to be quadratic
        self.state = tf.placeholder(tf.float32, shape=[None, input_dimensions[0], input_dimensions[1]])

        self.state_reshaped = tf.reshape(self.state, [-1, input_dimensions[0], input_dimensions[1], 1])

        self.keep_prob = tf.placeholder(tf.float32)

        self.W_conv1 = weight_variable([5, 5, 1, 32])
        self.b_conv1 = bias_variable([32])
        self.h_conv1 = tf.nn.relu(conv2d(self.state_reshaped, self.W_conv1) + self.b_conv1)
        self.h_pool1 = max_pool_2x2(self.h_conv1)

        self.W_conv2 = weight_variable([5, 5, 32, 64])
        self.b_conv2 = bias_variable([64])
        self.h_conv2 = tf.nn.relu(conv2d(self.h_pool1, self.W_conv2) + self.b_conv2)
        self.h_pool2 = max_pool_2x2(self.h_conv2)

        self.h_pool2_flat = tf.reshape(self.h_pool2, [-1, 10 * 10 * 64])

        # advantage
        self.W_fc_adv_1 = weight_variable([10 * 10 * 64, 250])
        self.b_fc_adv_1 = bias_variable([250])
        self.h_fc_adv_1 = tf.nn.relu(tf.matmul(self.h_pool2_flat, self.W_fc_adv_1) + self.b_fc_adv_1)
        self.h_fc_adv_1_drop = tf.nn.dropout(self.h_fc_adv_1, self.keep_prob)

        self.W_fc_adv_2 = weight_variable([250, num_actions])
        self.b_fc_adv_2 = bias_variable([num_actions])
        self.h_fc_adv_2 = tf.nn.tanh(tf.matmul(self.h_fc_adv_1_drop, self.W_fc_adv_2) + self.b_fc_adv_2)
        self.h_fc_adv_2_drop = tf.nn.dropout(self.h_fc_adv_2, self.keep_prob)

        # value
        self.W_fc_val_1 = weight_variable([10 * 10 * 64, 250])
        self.b_fc_val_1 = bias_variable([250])
        self.h_fc_val_1 = tf.nn.relu(tf.matmul(self.h_pool2_flat, self.W_fc_val_1) + self.b_fc_val_1)
        self.h_fc_val_1_drop = tf.nn.dropout(self.h_fc_val_1, self.keep_prob)

        self.W_fc_val_2 = weight_variable([250, 1])
        self.b_fc_val_2 = bias_variable([1])
        self.h_fc_val_2 = tf.nn.tanh(tf.matmul(self.h_fc_val_1_drop, self.W_fc_val_2) + self.b_fc_val_2)
        self.h_fc_val_2_drop = tf.nn.dropout(self.h_fc_val_2, self.keep_prob)

        self.Qout = self.h_fc_val_2_drop + tf.subtract(self.h_fc_adv_2_drop, reduce_mean(self.h_fc_adv_2_drop))

        self.predict = tf.argmax(self.Qout, axis=1)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, num_actions, dtype=tf.float32)

        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)

        self.targetQ = tf.placeholder(tf.float32, shape=[None, num_actions])

        self.td_error = tf.square(self.targetQ - self.Q)
        self.loss = tf.reduce_mean(self.td_error)

        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.updateModel = self.trainer.minimize(self.loss)


# v1
def model_dueling(input_dimensions, num_actions):
    graph = tf.Graph()

    with graph.as_default():
        # input assumed to be quadratic
        x = tf.placeholder(tf.float32, shape=[None, input_dimensions[0], input_dimensions[1]])
        y_ = tf.placeholder(tf.float32, shape=[None, 5])

        x_image = tf.reshape(x, [-1, input_dimensions[0], input_dimensions[1], 1])

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

        W_fc_adv_2 = weight_variable([250, num_actions])
        b_fc_adv_2 = bias_variable([num_actions])
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

        output = tf.subtract(tf.add(h_fc_adv_2_drop, h_fc_val_2_drop), reduce_mean(h_fc_adv_2_drop))

    return output


def model_dueling_keras(input_shape, output_dim):
    # Common Layers
    input_layer = Input(shape=input_shape)
    common_layer = Reshape(input_shape + (1,))(input_layer)
    common_layer = Conv2D(30, (5, 5), activation="relu")(common_layer)
    # common_layer = MaxPooling2D(pool_size=(2, 2))(common_layer)
    common_layer = Conv2D(15, (5, 5), activation="relu")(common_layer)
    # common_layer = MaxPooling2D(pool_size=(2, 2))(common_layer)
    common_layer = Flatten()(common_layer)

    common_layer = Dropout(.3)(common_layer)

    # adv_layer = Dropout(0.2)(common_layer)
    adv_layer = Dense(100, activation="relu")(common_layer)
    adv_layer = Dense(50, activation="relu")(adv_layer)
    adv_layer = Dense(output_dim, activation="tanh")(adv_layer)

    # val_layer = Dropout(0.2)(common_layer)
    val_layer = Dense(100, activation="relu")(common_layer)
    val_layer = Dense(50, activation="relu")(val_layer)
    val_layer = Dense(1, activation="linear")(val_layer)
    val_layer = RepeatVector(output_dim)(val_layer)
    val_layer = Flatten()(val_layer)
    # q = v + a - mean(a, reduction_indices=1, keep_dims=True)
    # q_layer = val_layer + adv_layer - reduce_mean(adv_layer, keep_dims=True)

    q_layer = merge(inputs=[adv_layer, val_layer], mode=lambda x: x[1] + x[0] - K.mean(x[0], keepdims=True),
                    output_shape=lambda x: x[0])
    # q_layer = Activation(activation="tanh")(q_layer)

    model = Model(inputs=[input_layer], outputs=[q_layer])

    model.compile(loss='mse',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


def model_acceleration_q(image_input_shape, velocity_input_shape,  num_outputs):

    #num_outputs = np.prod(output_dim)

    # this part of the network processes the
    img_input = Input(shape=image_input_shape)
    image_layer = Reshape(image_input_shape + (1,))(img_input)
    image_layer = Conv2D(30, (5, 5), activation="relu")(image_layer)
    # image_layer = MaxPooling2D(pool_size=(2, 2))(image_layer)
    image_layer = Conv2D(15, (5, 5), activation="relu")(image_layer)
    # image_layer = MaxPooling2D(pool_size=(2, 2))(image_layer)
    image_layer = Flatten()(image_layer)

    # this part takes care of the velocity input values
    vel_input = Input(shape=velocity_input_shape)
    vel_layer = Reshape(velocity_input_shape + (1,))(vel_input)
    vel_layer = Flatten()(vel_layer)

    # here, the preprocessed image and the velocities are merged into one tensor
    concat_layer = Concatenate()([image_layer, vel_layer])

    # advantage function of actions
    adv_layer = Dropout(.3)(concat_layer)
    adv_layer = Dense(100, activation="relu")(adv_layer)
    adv_layer = Dropout(.3)(adv_layer)
    adv_layer = Dense(50, activation="relu")(adv_layer)
    adv_layer = Dropout(.3)(adv_layer)
    adv_layer = Dense(25, activation="relu")(adv_layer)
    adv_layer = Dense(num_outputs, activation="tanh")(adv_layer)

    # value of state
    val_layer = Dropout(.3)(concat_layer)
    val_layer = Dense(100, activation="relu")(concat_layer)
    val_layer = Dropout(.3)(val_layer)
    val_layer = Dense(50, activation="relu")(val_layer)
    val_layer = Dropout(.3)(val_layer)
    val_layer = Dense(25, activation="relu")(val_layer)
    val_layer = Dense(1, activation="tanh")(val_layer)
    val_layer = RepeatVector(num_outputs)(val_layer)
    val_layer = Flatten()(val_layer)
    # q = v + a - mean(a, reduction_indices=1, keep_dims=True)
    # q_layer = val_layer + adv_layer - reduce_mean(adv_layer, keep_dims=True)

    # merging advantage function and state value
    q_layer = Lambda(lambda x: x[1] + x[0] - K.mean(x[0]))([adv_layer, val_layer])
    # q_layer = Activation(activation="tanh")(q_layer)

    #q_layer = Reshape(output_dim)(q_layer)

    model = Model(inputs=[img_input, vel_input], outputs=[q_layer])

    model.compile(loss='mse',
                  optimizer='adam',
                  metrics=['accuracy'])

    print(model.summary())

    return model


def model_acceleration_q_multidim(image_input_shape, velocity_input_shape, output_dim):
    num_outputs = np.prod(output_dim)

    # this part of the network processes the
    img_input = Input(shape=image_input_shape)
    image_layer = Reshape(image_input_shape + (1,))(img_input)
    image_layer = Conv2D(30, (5, 5), activation="relu")(image_layer)
    # image_layer = MaxPooling2D(pool_size=(2, 2))(image_layer)
    image_layer = Conv2D(15, (5, 5), activation="relu")(image_layer)
    # image_layer = MaxPooling2D(pool_size=(2, 2))(image_layer)
    image_layer = Flatten()(image_layer)

    # this part takes care of the velocity input values
    vel_input = Input(shape=velocity_input_shape)
    vel_layer = Reshape(velocity_input_shape + (1,))(vel_input)
    vel_layer = Flatten()(vel_layer)

    # here, the preprocessed image and the velocities are merged into one tensor
    concat_layer = Concatenate()([image_layer, vel_layer])

    # advantage function of actions
    adv_layer = Dropout(.3)(concat_layer)
    adv_layer = Dense(100, activation="relu")(adv_layer)
    adv_layer = Dropout(.3)(adv_layer)
    adv_layer = Dense(50, activation="relu")(adv_layer)
    adv_layer = Dense(num_outputs, activation="tanh")(adv_layer)

    # value of state
    val_layer = Dropout(.3)(concat_layer)
    val_layer = Dense(100, activation="relu")(concat_layer)
    val_layer = Dropout(.3)(val_layer)
    val_layer = Dense(50, activation="relu")(val_layer)
    val_layer = Dense(1, activation="tanh")(val_layer)
    val_layer = RepeatVector(num_outputs)(val_layer)
    val_layer = Flatten()(val_layer)
    # q = v + a - mean(a, reduction_indices=1, keep_dims=True)
    # q_layer = val_layer + adv_layer - reduce_mean(adv_layer, keep_dims=True)

    # merging advantage function and state value
    q_layer = Lambda(lambda x: x[1] + x[0] - K.mean(x[0]))([adv_layer, val_layer])
    # q_layer = Activation(activation="tanh")(q_layer)

    q_layer = Reshape(output_dim)(q_layer)

    model = Model(inputs=[img_input, vel_input], outputs=[q_layer])

    model.compile(loss='mse',
                  optimizer='adam',
                  metrics=['accuracy'])

    print(model.summary())

    return model

def action_cascade_network(image_input_shape, velocity_input_shape, action_dim):
    # num_outputs = np.prod(output_dim)

    train_network = Input(shape=(1,))

    train_action = Input(shape=action_dim)

    action_0 = Lambda(lambda x: x[:, 0, :])(train_action)
    action_1 = Lambda(lambda x: x[:, 1, :])(train_action)

    # action_0 = Input(shape=(action_dim,))
    # action_1 = Input(shape=(action_dim,))
    # #action_2 = Input(shape=(action_dim,))

    # this part of the network processes the
    img_input = Input(shape=image_input_shape)
    image_layer = Reshape(image_input_shape + (1,))(img_input)
    image_layer = Conv2D(30, (5, 5), activation="relu")(image_layer)
    # image_layer = MaxPooling2D(pool_size=(2, 2))(image_layer)
    image_layer = Conv2D(15, (5, 5), activation="relu")(image_layer)
    # image_layer = MaxPooling2D(pool_size=(2, 2))(image_layer)
    image_layer = Flatten()(image_layer)

    # this part takes care of the velocity input values
    vel_input = Input(shape=velocity_input_shape)
    vel_layer = Reshape(velocity_input_shape + (1,))(vel_input)
    vel_layer = Flatten()(vel_layer)

    # here, the preprocessed image and the velocities are merged into one tensor
    feature_layer = Concatenate()([image_layer, vel_layer])
    feature_layer = Dropout(.2)(feature_layer)
    
    # PART 1

    # advantage function of actions
    part_1_adv_layer = Dense(100, activation="relu")(feature_layer)
    part_1_adv_layer = Dropout(.2)(part_1_adv_layer)
    part_1_adv_layer = Dense(50, activation="relu")(part_1_adv_layer)
    part_1_adv_layer = Dropout(.2)(part_1_adv_layer)
    part_1_adv_layer = Dense(25, activation="relu")(part_1_adv_layer)
    part_1_adv_layer = Dropout(.2)(part_1_adv_layer)
    part_1_adv_layer = Dense(action_dim[0], activation="tanh")(part_1_adv_layer)


    # value of state
    part_1_val_layer = Dense(100, activation="relu")(feature_layer)
    part_1_val_layer = Dropout(.2)(part_1_val_layer)
    part_1_val_layer = Dense(50, activation="relu")(part_1_val_layer)
    part_1_val_layer = Dropout(.2)(part_1_val_layer)
    part_1_val_layer = Dense(25, activation="relu")(part_1_val_layer)
    part_1_val_layer = Dropout(.2)(part_1_val_layer)
    part_1_val_layer = Dense(1, activation="linear")(part_1_val_layer)
    part_1_val_layer = RepeatVector(action_dim[0])(part_1_val_layer)
    part_1_val_layer = Flatten()(part_1_val_layer)

    # merging advantage function and state value
    q_layer_action_0 = Lambda(lambda x: x[1] + x[0] - K.mean(x[0]))([part_1_adv_layer, part_1_val_layer])

        #merge(inputs=[part_1_adv_layer, part_1_val_layer], mode=lambda x: x[1] + x[0] - K.mean(x[0], keepdims=True),
        #            output_shape=lambda x: x[0])

    part_1_action_0 = Activation('softmax')(q_layer_action_0)

    part_1_action_0 = Lambda(lambda x: x[0]*x[1] + (1-x[0])*x[2])([train_network, action_0, part_1_action_0])
    
    # PART 2

    part_2_input = Concatenate()([part_1_action_0, feature_layer])

    # advantage function of actions
    part_2_adv_layer = Dense(100, activation="relu")(part_2_input)
    part_2_adv_layer = Dropout(.2)(part_2_adv_layer)
    part_2_adv_layer = Dense(50, activation="relu")(part_2_adv_layer)
    part_2_adv_layer = Dropout(.2)(part_2_adv_layer)
    part_2_adv_layer = Dense(action_dim[0], activation="tanh")(part_2_adv_layer)

    # value of state
    part_2_val_layer = Dense(100, activation="relu")(part_2_input)
    part_2_val_layer = Dropout(.2)(part_2_val_layer)
    part_2_val_layer = Dense(50, activation="relu")(part_2_val_layer)
    part_2_val_layer = Dropout(.2)(part_2_val_layer)
    part_2_val_layer = Dense(1, activation="linear")(part_2_val_layer)
    part_2_val_layer = RepeatVector(action_dim[0])(part_2_val_layer)
    part_2_val_layer = Flatten()(part_2_val_layer)

    # merging advantage function and state value
    q_layer_action_1 = Lambda(lambda x: x[1] + x[0] - K.mean(x[0]))([part_2_adv_layer, part_2_val_layer])

    part_2_action_1 = Activation('softmax')(q_layer_action_1)

    part_2_action_1 = Lambda(lambda x: x[0] * x[1] + (1 - x[0]) * x[2])([train_network, action_1, part_2_action_1])

    # PART 3

    part_3_input = Concatenate()([part_2_action_1, feature_layer])

    # advantage function of actions
    part_3_adv_layer = Dense(100, activation="relu")(part_3_input)
    part_3_adv_layer = Dropout(.2)(part_3_adv_layer)
    part_3_adv_layer = Dense(50, activation="relu")(part_3_adv_layer)
    part_3_adv_layer = Dropout(.2)(part_3_adv_layer)
    part_3_adv_layer = Dense(action_dim[0], activation="tanh")(part_3_adv_layer)

    # value of state
    part_3_val_layer = Dense(100, activation="relu")(part_3_input)
    part_3_val_layer = Dropout(.2)(part_3_val_layer)
    part_3_val_layer = Dense(50, activation="relu")(part_3_val_layer)
    part_3_val_layer = Dropout(.2)(part_3_val_layer)
    part_3_val_layer = Dense(1, activation="linear")(part_3_val_layer)
    part_3_val_layer = RepeatVector(action_dim[0])(part_3_val_layer)
    part_3_val_layer = Flatten()(part_3_val_layer)

    # merging advantage function and state value
    q_layer_action_2 = Lambda(lambda x: x[1] + x[0] - K.mean(x[0]))([part_3_adv_layer, part_3_val_layer])

    part_3_action_2 = Activation('softmax')(q_layer_action_2)

    #part_3_action_2 = Lambda(lambda x: x[0] * x[1] + (1 - x[0]) * x[2])([train_network, action_2, part_3_action_2])

    part_1_action_0 = Reshape((1, 3))(part_1_action_0)
    part_2_action_1 = Reshape((1, 3))(part_2_action_1)
    part_3_action_2 = Reshape((1, 3))(part_3_action_2)

    pred_action = Concatenate(axis=1)([part_1_action_0, part_2_action_1, part_3_action_2])

    model = Model(inputs=[img_input, vel_input, train_network, train_action],
                  outputs=[pred_action])

    model.compile(loss='mse',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


if __name__ == "__main__":

    model = action_cascade_network((40,40), (3,), (3,3))

    print(model.summary())

    for i in range(5):
        X = np.random.random((100, 40, 40)).astype('float32')
        V = np.random.random((100,3)).astype('float32')
        y = np.random.random((100, 5)).astype('float32')
        train = np.ones(100)
        action = np.random.random((100,3,3))

        model.fit(x=[X,V,train,action], y=[action])

    for i in range(5):
        X = np.random.random((1, 40, 40)).astype('float32')
        V = np.random.random((1,3)).astype('float32')
        train = np.zeros(100)
        action = np.zeros((100, 3, 3))
        y = model.predict(x=[X,V,train,action])
        print(np.argmax(y, axis=0))

    # print(model_dueling_keras((40, 40), 5).summary())
    #
    # model = model_acceleration_q((40, 40), (3,), 5)
    #
    # X = np.random.random((10, 40, 40)).astype('float32')
    # V = np.random.random((10,3)).astype('float32')
    #
    # # y = np.random.random((100, 6, 1))
    # # print(X.shape)
    #
    # # X = X.reshape(len(X), 1, 40, 40).astype('float32')
    #
    # print(model.summary())
    #
    # for i in range(5):
    #     X = np.random.random((100, 40, 40)).astype('float32')
    #     V = np.random.random((100,3)).astype('float32')
    #     y = np.random.random((100, 5)).astype('float32')
    #     model.fit(x=[X,V], y=y)
    #
    # for i in range(5):
    #     X = np.random.random((1, 40, 40)).astype('float32')
    #     V = np.random.random((1,3)).astype('float32')
    #     y = model.predict(x=[X,V])
    #     print(y.argmax(axis=1))


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
