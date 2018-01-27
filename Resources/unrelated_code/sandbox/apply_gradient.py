from keras.models import Sequential
import keras.models
from keras.layers import Reshape, Conv2D, Flatten, Dense
import numpy as np
import tensorflow as tf
from keras import backend as K

def create_model(input_shape, output_dim):
    model = Sequential()

    model.add(Reshape(input_shape + (1,), input_shape=input_shape))
    model.add(Conv2D(30, kernel_size=(5, 5), activation="relu"))
    model.add(Flatten())
    model.add(Dense(50, activation="relu"))
    model.add(Dense(output_dim, activation="tanh"))

    model.compile(loss='mse',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

input_dim = (10,10)
output_num = 3

actor = create_model(input_dim, output_num)
# print(actor.predict(np.ones((1, 10,10))))
#
batch_states = np.random.random((10,10,10))
batch_actions = np.random.random((10,3))

actor.train_on_batch(batch_states, batch_actions)
#
# print(actor.predict(np.ones((1, 10,10))))

actor.save('actor.h5')
actor = keras.models.load_model('actor.h5')

# print(actor.predict(np.ones((1, 10,10))))
#
# batch_states = np.random.random((10,10,10))
# batch_actions = np.random.random((10,3))
#
# actor.train_on_batch(batch_states, batch_actions)
#
# print(actor.predict(np.ones((1, 10,10))))

print(actor.output)
print(actor.trainable_weights)
print(tf.trainable_variables())

action_gradient = tf.placeholder(tf.float32, actor.output_shape)
params_grad = tf.gradients(actor.output, actor.trainable_weights, -action_gradient)
grads = zip(params_grad, actor.trainable_weights)
optimize = tf.train.AdamOptimizer().apply_gradients(grads)

sess = K.get_session()
#sess = tf.Session()

global_vars = tf.global_variables()
is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

print([str(i.name) for i in not_initialized_vars]) # only for testing
if len(not_initialized_vars):
    sess.run(tf.variables_initializer(not_initialized_vars))


print(actor.predict(np.ones((1, 10,10))))

#print([print(i) for i in actor.layers])

batch_states = np.random.random((10,10,10))
action_grads = np.random.random((10,3))

sess.run(optimize, feed_dict={
    actor.inputs[0]: batch_states,
    action_gradient: action_grads
})

print(actor.predict(np.ones((1, 10,10))))