from keras.models import Sequential, Model
from keras.layers import Input, Dense, Concatenate, Reshape, Conv2D, MaxPooling2D, Dropout, Flatten, RepeatVector, \
    Lambda
import keras.backend as K
from keras.initializers import RandomUniform


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
    # common_layer = MaxPooling2D(pool_size=(2, 2))(common_layer)
    common_layer = Conv2D(15, (5, 5), activation="relu", name='conv_2')(common_layer)
    # common_layer = MaxPooling2D(pool_size=(2, 2))(common_layer)
    common_layer = Flatten()(common_layer)

    # adv_layer = Dropout(0.2)(common_layer)
    adv_layer = Dropout(.3)(common_layer)
    adv_layer = Dense(100, activation="relu", name='adv_1')(adv_layer)
    adv_layer = Dropout(.3)(adv_layer)
    adv_layer = Dense(50, activation="relu", name='adv_2')(adv_layer)
    adv_layer = Dense(output_dim, activation="tanh", name='adv_3')(adv_layer)

    # val_layer = Dropout(0.2)(common_layer)
    val_layer = Dropout(.3)(common_layer)
    val_layer = Dense(100, activation="relu", name='val_1')(val_layer)
    val_layer = Dropout(.3)(val_layer)
    val_layer = Dense(50, activation="relu", name='val_2')(val_layer)
    val_layer = Dense(1, activation="tanh", name='val_3')(val_layer)
    val_layer = RepeatVector(output_dim)(val_layer)
    val_layer = Flatten()(val_layer)
    # q = v + a - mean(a, reduction_indices=1, keep_dims=True)
    # q_layer = val_layer + adv_layer - reduce_mean(adv_layer, keep_dims=True)

    q_layer = Lambda(lambda x: x[1] + x[0] - K.mean(x[0]), name='q_layer')([adv_layer, val_layer])

    model = Model(inputs=[input_layer], outputs=[q_layer])

    model.compile(loss='mse',
                  optimizer='adam',
                  metrics=['accuracy'])

    # print(model.summary())

    return model


def actor_network(input_shape):
    state_input = Input(shape=input_shape)
    conv_module = Reshape(input_shape + (1,))(state_input)
    conv_module = Conv2D(30, (3, 3), activation="relu", name='conv_1')(conv_module)
    conv_module = Conv2D(30, (3, 3), activation="relu", name='conv_2')(conv_module)
    conv_module = Conv2D(30, (3, 3), activation="relu", name='conv_3')(conv_module)
    conv_module = Flatten()(conv_module)

    #mlp_module = Dropout(.2)(conv_module)
    mlp_module = Dense(100, activation="relu", name='mlp_1')(conv_module)
    #mlp_module = Dropout(.2)(mlp_module)
    mlp_module = Dense(100, activation="relu", name='mlp_2')(mlp_module)
    #mlp_module = Dropout(.2)(mlp_module)

    forward = Dense(1, activation="sigmoid", name='forward', kernel_initializer=RandomUniform(-.0003, .0003))(mlp_module)
    sideways = Dense(1, activation="tanh", name='sideways', kernel_initializer=RandomUniform(-.0003, .0003))(mlp_module)
    rotation = Dense(1, activation="tanh", name='rotation', kernel_initializer=RandomUniform(-.0003, .0003))(mlp_module)

    action_output = Concatenate()([forward, sideways, rotation])

    model = Model(inputs=[state_input], outputs=[action_output])

    model.compile(loss='mse',
                  optimizer='adam',
                  metrics=['accuracy'])

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
    m = model_dueling((50, 50), 5)
    print(m.summary())
    pass
