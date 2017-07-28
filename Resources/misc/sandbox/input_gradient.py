from keras.models import Model
from keras.layers import Input, Dense, Concatenate
import numpy as np
import keras.backend as K
import tensorflow as tf


def simple_model():
    a = Input(shape=(3,), name='a')
    b = Input(shape=(5,), name='b')

    concat = Concatenate()([a,b])
    c = Dense(1)(concat)
    model = Model(inputs=[a,b], outputs=c)

    model.compile(loss='mse',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


model = simple_model()

print(model.get_weights())

# print(model.inputs[0])
sess = tf.Session()

critic_action_grads = tf.gradients(model.outputs, model.inputs[1])  #GRADIENTS for policy update
sess.run(tf.global_variables_initializer())

action_grads = sess.run(critic_action_grads, feed_dict={
            model.inputs[0]: np.ones((1,3)),
            model.inputs[1]: np.ones((1,5))
        })[0]

print(action_grads)

input()


gradients = K.gradients(model.outputs, model.inputs[0])[0]  # gradient tensors
get_gradients = K.function(inputs=model.inputs, outputs=[gradients])

print(get_gradients([np.ones((1,3)), np.ones((1,5))]))

model.train_on_batch(x=[np.ones((10,3)), np.ones((10,5))], y=np.ones(10)*.5)

print(model.get_weights())
print(get_gradients([np.ones((1,3)), np.ones((1,5))]))