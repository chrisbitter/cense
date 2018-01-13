from keras.models import Model
from keras.layers import Input, Dense, Concatenate, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.initializers import RandomUniform
import numpy as np

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
    conv_module = Dropout(.2)(conv_module)

    mlp_module = Dense(400, activation='relu', name='dense_1')(conv_module)
    mlp_module = Dropout(.2)(mlp_module)
    mlp_module = Dense(200, activation='relu', name='dense_2')(mlp_module)
    mlp_module = Dropout(.2)(mlp_module)
    mlp_module = Dense(100, activation='relu', name='dense_3')(mlp_module)
    mlp_module = Dropout(.2)(mlp_module)

    forward_action = Dense(1, activation='sigmoid', name='forward')(mlp_module)
    sideways_action = Dense(1, activation='tanh', name='sideways')(mlp_module)
    rotation_action = Dense(1, activation='tanh', name='rotation')(mlp_module)

    action_output = Concatenate()([forward_action, sideways_action, rotation_action])

    model = Model(inputs=[state_input], outputs=[action_output])

    model.compile(loss='mse',
                  optimizer='adam')

    return model


if __name__ == "__main__":

    shape = (40, 40, 3)

    m = actor_network(shape)
    print(m.summary())

    print(m.predict(np.random.random((1,)+shape)*2-1))
