from keras.models import Sequential, Model
from keras.layers import Input, Dense, Concatenate, Reshape, Conv2D, MaxPooling2D, Dropout, Flatten, RepeatVector, \
    Lambda
import keras.backend as K
from keras.initializers import RandomUniform
from keras.layers.advanced_activations import PReLU
from keras.regularizers import l2
import numpy as np

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


def model_dueling(input_shape, output_dim):
    # Common Layers
    input_layer = Input(shape=input_shape)
    common_layer = Reshape(input_shape + (1,))(input_layer)
    common_layer = Conv2D(30, (5, 5), activation="relu", name='conv_1')(common_layer)
    common_layer = MaxPooling2D(pool_size=(2, 2))(common_layer)
    common_layer = Conv2D(15, (5, 5), activation="relu", name='conv_2')(common_layer)
    common_layer = MaxPooling2D(pool_size=(2, 2))(common_layer)
    common_layer = Flatten()(common_layer)

    adv_layer = Dropout(.2)(common_layer)
    adv_layer = Dense(200, activation="relu", name='adv_1')(adv_layer)
    adv_layer = Dropout(.2)(adv_layer)
    adv_layer = Dense(200, activation="relu", name='adv_2')(adv_layer)
    adv_layer = Dropout(.2)(adv_layer)
    adv_layer = Dense(output_dim, activation="linear", name='adv_3')(adv_layer)

    val_layer = Dropout(.2)(common_layer)
    val_layer = Dense(200, activation="relu", name='val_1')(val_layer)
    val_layer = Dropout(.2)(val_layer)
    val_layer = Dense(200, activation="relu", name='val_2')(val_layer)
    val_layer = Dropout(.2)(val_layer)
    val_layer = Dense(1, activation="linear", name='val_3')(val_layer)
    val_layer = RepeatVector(output_dim)(val_layer)
    val_layer = Flatten()(val_layer)

    q_layer = Lambda(lambda x: x[1] + x[0] - K.mean(x[0]), name='q_layer')([adv_layer, val_layer])

    model = Model(inputs=[input_layer], outputs=[q_layer])

    model.compile(loss='mse',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model

def lstm(input_dim, output_dim):

    state_input = Input(shape=input_dim)

    #mlp_module = Reshape(input_dim + (1,))(state_input)
    mlp_module = Flatten()(state_input)
    mlp_module = Dense(np.prod(output_dim))(mlp_module)

    action_output = Reshape(output_dim)(mlp_module)

    model = Model(inputs=[state_input], outputs=[action_output])

    model.compile(loss='mse',
                  optimizer='adam',
                  metrics=['accuracy'])

    print(model.summary())

    return model


def actor_network(input_shape):
    state_input = Input(shape=input_shape)
    # conv_module = Reshape(input_shape + (1,))(state_input)
    conv_module = Conv2D(30, (5, 5), activation="relu", name='conv_1')(state_input)  # 36x36x30
    conv_module = MaxPooling2D(pool_size=(2, 2))(conv_module)  # 18x18x30
    # conv_module = Dropout(.5)(conv_module)
    conv_module = Conv2D(15, (5, 5), activation="relu", name='conv_2')(conv_module)  # 14x14x30
    conv_module = MaxPooling2D(pool_size=(2, 2))(conv_module)  # 7x7x15
    # conv_module = Dropout(.5)(conv_module)
    conv_module = Conv2D(10, (3, 3), activation="relu", name='conv_3')(conv_module)  # 5x5x10
    # conv_module = MaxPooling2D(pool_size=(2, 2))(conv_module)  # 7x7x15
    # conv_module = Dropout(.5)(conv_module)
    conv_module = Flatten()(conv_module)
    # conv_module = Dropout(.2)(conv_module)

    mlp_module = Dense(200, activation='relu', name='mlp_1')(conv_module)
    mlp_module = Dropout(.2)(mlp_module)
    # mlp_module = Dense(100, activation='relu', name='mlp_2')(mlp_module)
    # mlp_module = Dropout(.2)(mlp_module)

    forward = Dense(200, activation='relu', name='forward_1')(mlp_module)
    forward = Dropout(.2)(forward)
    # forward = Dense(100, activation='relu', name='forward_.5')(forward)
    # forward = Dropout(.2)(forward)
    forward = Dense(1, activation="sigmoid", name='forward_2', kernel_initializer=RandomUniform(-.0003, .0003))(
        forward)

    sideways = Dense(200, activation='relu', name='sideways_1')(mlp_module)
    sideways = Dropout(.2)(sideways)
    # sideways = Dense(100, activation='relu', name='sideways_.5')(sideways)
    # sideways = Dropout(.2)(sideways)
    sideways = Dense(1, activation="tanh", name='sideways_2', kernel_initializer=RandomUniform(-.0003, .0003))(sideways)

    rotation = Dense(200, activation='relu', name='rotation_1')(mlp_module)
    rotation = Dropout(.2)(rotation)
    # rotation = Dense(100, activation='relu', name='rotation_.5')(rotation)
    # rotation = Dropout(.2)(rotation)
    rotation = Dense(1, activation="tanh", name='rotation_2', kernel_initializer=RandomUniform(-.0003, .0003))(rotation)

    action_output = Concatenate()([forward, sideways, rotation])

    model = Model(inputs=[state_input], outputs=[action_output])

    model.compile(loss='mse',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


def actor_network_(input_shape):
    state_input = Input(shape=input_shape)
    conv_module = Reshape(input_shape + (1,))(state_input)
    conv_module = Conv2D(30, (5, 5), activation="relu", name='conv_1')(conv_module)  # 36x36x30
    conv_module = MaxPooling2D(pool_size=(2, 2))(conv_module)   # 18x18x30
    # conv_module = Dropout(.5)(conv_module)
    # common_layer = MaxPooling2D(pool_size=(2, 2))(common_layer)
    conv_module = Conv2D(15, (5, 5), activation="relu", name='conv_2')(conv_module)  # 14x14x30
    conv_module = MaxPooling2D(pool_size=(2, 2))(conv_module)  # 7x7x15
    # conv_module = Dropout(.5)(conv_module)
    conv_module = Conv2D(10, (3, 3), activation="relu", name='conv_3')(conv_module)  # 5x5x10
    #conv_module = MaxPooling2D(pool_size=(2, 2))(conv_module)  # 7x7x15
    # conv_module = Dropout(.5)(conv_module)
    conv_module = Flatten()(conv_module)

    mlp_module = Dense(200, activation='relu', name='mlp_1')(conv_module)
    # mlp_module = Dropout(.5)(mlp_module)
    mlp_module = Dense(200, activation='relu', name='mlp_2')(mlp_module)

    #forward = Dense(200, activation='relu', name='forward_1')(mlp_module)
    # forward = Dropout(.5)(forward)
    forward = Dense(1, activation="sigmoid", name='forward', kernel_initializer=RandomUniform(-.0003, .0003))(
        mlp_module)

    #sideways = Dense(200, activation='relu', name='sideways_1')(mlp_module)
    # sideways = Dropout(.5)(sideways)
    sideways = Dense(1, activation="tanh", name='sideways', kernel_initializer=RandomUniform(-.0003, .0003))(mlp_module)

    #rotation = Dense(200, activation='relu', name='rotation_1')(mlp_module)
    # rotation = Dropout(.5)(rotation)
    rotation = Dense(1, activation="tanh", name='rotation', kernel_initializer=RandomUniform(-.0003, .0003))(mlp_module)

    action_output = Concatenate()([forward, sideways, rotation])

    model = Model(inputs=[state_input], outputs=[action_output])

    model.compile(loss='mse',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


def actor_network_2(input_shape):
    state_input = Input(shape=input_shape)
    conv_module = Reshape(input_shape + (1,))(state_input)
    conv_module = Conv2D(32, (5, 5), name='conv_1')(conv_module)    # 56x56x32
    conv_module = PReLU(name="prelu_1")(conv_module)
    conv_module = MaxPooling2D()(conv_module)                       # 28x28x32
    # conv_module = Dropout(.5)(conv_module)
    conv_module = Conv2D(16, (5, 5), name='conv_2')(conv_module)    # 24x24x16
    conv_module = MaxPooling2D()(conv_module)                       # 12x12x16
    conv_module = PReLU(name="prelu_2")(conv_module)
    # conv_module = Dropout(.5)(conv_module)
    conv_module = Conv2D(8, (3, 3), name='conv_3')(conv_module)     # 10x10x8
    conv_module = MaxPooling2D()(conv_module)                       # 5x5x8
    conv_module = PReLU(name="prelu_3")(conv_module)
    #conv_module = Conv2D(30, (3, 3), activation='elu', name='conv_3')(conv_module)
    conv_module = Flatten()(conv_module)
    conv_module = Dropout(.5)(conv_module)

    #mlp_module = Dropout(.2)(conv_module)
    mlp_module = Dense(100, activation='relu', name='mlp_1')(conv_module)
    mlp_module = PReLU(name="prelu_4")(mlp_module)
    mlp_module = Dropout(.4)(mlp_module)
    mlp_module = Dense(100, activation='relu', name='mlp_2')(mlp_module)
    mlp_module = PReLU(name="prelu_5")(mlp_module)
    mlp_module = Dropout(.4)(mlp_module)

    forward = Dense(1, activation="sigmoid", name='forward', kernel_initializer=RandomUniform(-.0003, .0003))(mlp_module)
    sideways = Dense(1, activation="tanh", name='sideways', kernel_initializer=RandomUniform(-.0003, .0003))(mlp_module)
    rotation = Dense(1, activation="tanh", name='rotation', kernel_initializer=RandomUniform(-.0003, .0003))(mlp_module)

    action_output = Concatenate()([forward, sideways, rotation])

    model = Model(inputs=[state_input], outputs=[action_output])

    model.compile(loss='mse',
                  optimizer='adam',
                  metrics=['accuracy'])

    # model2 = Model(inputs=[state_input], outputs=[mlp_module])
    #
    # model2.compile(loss='mse',
    #               optimizer='adam',
    #               metrics=['accuracy'])
    #
    # return model , model2

    return model


def critic_network(input_shape):
    state_input = Input(shape=input_shape)
    action_input = Input(shape=(3,))

    conv_module = Reshape(input_shape + (1,))(state_input)
    conv_module = Conv2D(30, (3, 3), activation="relu", name='conv_1')(conv_module)
    conv_module = Conv2D(30, (3, 3), activation="relu", name='conv_2')(conv_module)
    conv_module = Conv2D(30, (3, 3), activation="relu", name='conv_3')(conv_module)
    conv_module = Flatten()(conv_module)

    features = Concatenate()(conv_module, action_input)

    mlp_module = Dropout(.3)(features)
    mlp_module = Dense(100, activation="relu", name='mlp_1')(mlp_module)
    mlp_module = Dropout(.3)(mlp_module)
    mlp_module = Dense(50, activation="relu", name='mlp_2')(mlp_module)

    q_value_output = Dense(1, activation="tanh", name='q_value')(mlp_module)

    model = Model(inputs=[state_input, action_input], outputs=[q_value_output])

    model.compile(loss='mse',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


if __name__ == "__main__":
    m = actor_network((40, 40))
    print(m.summary())
