from keras.models import Model
from keras.layers import Input, Dense, Concatenate, Conv2D, Dropout, Flatten, Reshape, MaxPooling2D, Lambda, RepeatVector
from keras.initializers import RandomUniform
from keras.regularizers import l2
#from keras.layers.advanced_activations import ELU
from keras.layers.advanced_activations import PReLU
import keras.backend as K

    
def critic_network(input_shape):
    state_input = Input(shape=input_shape)
    action_input = Input(shape=(3,))

    #conv_module = Reshape(input_shape + (1,))(state_input)
    conv_module = Conv2D(30, (5, 5), activation="relu", name='conv_1')(state_input)  # 36x36x30
    conv_module = MaxPooling2D(pool_size=(2, 2))(conv_module)   # 18x18x30
    #conv_module = Dropout(.5)(conv_module)
    # common_layer = MaxPooling2D(pool_size=(2, 2))(common_layer)
    conv_module = Conv2D(15, (5, 5), activation="relu", name='conv_2')(conv_module)  # 14x14x30
    conv_module = MaxPooling2D(pool_size=(2, 2))(conv_module)  # 7x7x15
    #conv_module = Dropout(.5)(conv_module)
    conv_module = Conv2D(10, (3, 3), activation="relu", name='conv_3')(conv_module)  # 5x5x10
    #conv_module = MaxPooling2D(pool_size=(2, 2))(conv_module)  # 7x7x15
    #conv_module = Dropout(.5)(conv_module)
    conv_module = Flatten()(conv_module)

    mlp_module = Dense(200, activation='relu', name='mlp_1')(conv_module)
    #mlp_module = Dropout(.2)(mlp_module)
    mlp_module = Concatenate()([mlp_module, action_input])
    mlp_module = Dense(200, activation='relu', name='mlp_2')(mlp_module)
    #mlp_module = Dropout(.2)(mlp_module)
    
    q_value_output = Dense(1, activation="linear", name='critic_q_value', kernel_initializer=RandomUniform(-.0003, .0003))(mlp_module)
    #q_value_output = Lambda(lambda x: 2*x, name='scaled_q_value')(q_value_output)

    model = Model(inputs=[state_input, action_input], outputs=[q_value_output])

    model.compile(loss='mse',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model
    
def actor_network(input_shape):
    state_input = Input(shape=input_shape)
    #conv_module = Reshape(input_shape + (1,))(state_input)
    conv_module = Conv2D(30, (5, 5), activation="relu", name='conv_1')(state_input)  # 36x36x30
    conv_module = MaxPooling2D(pool_size=(2, 2))(conv_module)  # 18x18x30
    # conv_module = Dropout(.5)(conv_module)
    # common_layer = MaxPooling2D(pool_size=(2, 2))(common_layer)
    conv_module = Conv2D(15, (5, 5), activation="relu", name='conv_2')(conv_module)  # 14x14x30
    conv_module = MaxPooling2D(pool_size=(2, 2))(conv_module)  # 7x7x15
    # conv_module = Dropout(.5)(conv_module)
    conv_module = Conv2D(10, (3, 3), activation="relu", name='conv_3')(conv_module)  # 5x5x10
    # conv_module = MaxPooling2D(pool_size=(2, 2))(conv_module)  # 7x7x15
    # conv_module = Dropout(.5)(conv_module)
    conv_module = Flatten()(conv_module)
    #conv_module = Dropout(.2)(conv_module)

    mlp_module = Dense(200, activation='relu', name='mlp_1')(conv_module)
    mlp_module = Dropout(.2)(mlp_module)

    forward = Dense(200, activation='relu', name='forward_1')(mlp_module)
    forward = Dropout(.2)(forward)
    forward = Dense(1, activation="sigmoid", name='forward_2', kernel_initializer=RandomUniform(-.0003, .0003))(
        forward)

    sideways = Dense(200, activation='relu', name='sideways_1')(mlp_module)
    sideways = Dropout(.2)(sideways)
    sideways = Dense(1, activation="tanh", name='sideways_2', kernel_initializer=RandomUniform(-.0003, .0003))(sideways)

    rotation = Dense(200, activation='relu', name='rotation_1')(mlp_module)
    rotation = Dropout(.2)(rotation)
    rotation = Dense(1, activation="tanh", name='rotation_2', kernel_initializer=RandomUniform(-.0003, .0003))(rotation)

    action_output = Concatenate()([forward, sideways, rotation])

    model = Model(inputs=[state_input], outputs=[action_output])

    model.compile(loss='mse',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model
    
    
# v1
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
    # q = v + a - mean(a, reduction_indices=1, keep_dims=True)
    # q_layer = val_layer + adv_layer - reduce_mean(adv_layer, keep_dims=True)

    q_layer = Lambda(lambda x: x[1] + x[0] - K.mean(x[0]), name='q_layer')([adv_layer, val_layer])

    model = Model(inputs=[input_layer], outputs=[q_layer])

    model.compile(loss='mse',
                  optimizer='adam',
                  metrics=['accuracy'])

    # print(model.summary())

    return model