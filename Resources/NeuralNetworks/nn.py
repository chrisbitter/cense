from keras.models import Sequential, Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten, RepeatVector, merge, Activation
from keras.layers.merge import Average, Add
from keras.backend import mean

import numpy as np


def model_simple_conv(input_dim, output_dim):
    model = Sequential()

    model.add(Conv2D(30, kernel_size=(5, 5), input_shape=input_dim, activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(15, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dense(50, activation="relu"))
    model.add(Dense(output_dim, activation="softmax"))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


def model_dueling(input_dim, output_dim):
    # Common Layers
    input_layer = Input(shape=input_dim)
    common_layer = Conv2D(30, (5, 5), input_shape=input_dim, activation="relu")(input_layer)
    common_layer = MaxPooling2D(pool_size=(2, 2))(common_layer)
    common_layer = Conv2D(15, (3, 3), activation="relu")(common_layer)
    common_layer = MaxPooling2D(pool_size=(2, 2))(common_layer)
    common_layer = Flatten()(common_layer)

    adv_layer = Dropout(0.2)(common_layer)
    adv_layer = Dense(128, activation="relu")(adv_layer)
    adv_layer = Dense(50, activation="relu")(adv_layer)
    adv_layer = Dense(output_dim, activation="tanh")(adv_layer)

    val_layer = Dropout(0.2)(common_layer)
    val_layer = Dense(128, activation="relu")(val_layer)
    val_layer = Dense(50, activation="relu")(val_layer)
    val_layer = Dense(1, activation="linear")(val_layer)
    val_layer = RepeatVector(output_dim)(val_layer)
    val_layer = Flatten()(val_layer)
    # q = v + a - mean(a, reduction_indices=1, keep_dims=True)
    merge_layer = merge(inputs=[adv_layer, val_layer], mode=lambda x: x[1] + x[0] - mean(x[0], keepdims=True),
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

    #y = np.random.random((100, 6, 1))
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