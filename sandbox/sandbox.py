import numpy as np
import h5py
# import Cense.NeuralNetworkFactory.nnFactory as factory
from Cense.World.Camera.camera_videocapture import Camera
import matplotlib.pyplot as plt
import time
from pyfirmata import Arduino, util
import Cense.NeuralNetworkFactory.nnFactory as factory
import tensorflow as tf
import os


def save_array():
    states = []
    actions = []
    rewards = []
    suc_states = []
    terminals = []

    experience = []

    for i in range(50):
        state = np.random.rand(50, 50)
        action = np.random.rand(6, 1)
        reward = np.random.uniform(-10, 10)
        suc_state = np.random.rand(50, 50)
        terminal = np.random.randint(2)

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        suc_states.append(suc_state)
        terminals.append(terminal)

    f = h5py.File('dummy_data.h5', 'w')

    f.create_dataset('states', data=states)
    f.create_dataset('actions', data=actions)
    f.create_dataset('rewards', data=rewards)
    f.create_dataset('suc_states', data=suc_states)
    f.create_dataset('terminals', data=terminals)
    f.close()


def load_array():
    f = h5py.File('dummy_data.h5', 'r')

    data = f['states'][:]
    print(data.shape)

    f.close()


def save_model():
    model = factory.model_simple_conv((50, 50), 6)

    with open('model.json', 'w') as file:
        file.write(model.to_json())

    model.save_weights("weights.h5")


def read_arduino():
    board = Arduino('com3')

    it = util.Iterator(board)
    it.start()

    board.analog[0].enable_reporting()
    board.analog[1].enable_reporting()

    analog_0 = board.get_pin('a:0:i')
    analog_1 = board.get_pin('a:1:i')

    while True:

        value_a0 = analog_0.read()
        value_a1 = analog_1.read()

        if value_a0 is not None and value_a1 is not None:

            if value_a0 > value_a1:
                board.digital[13].write(1)

            else:
                board.digital[13].write(0)


def stream_webcam():
    cam = Camera()

    fig = plt.figure(0)
    plt.ion()
    plt.imshow(np.zeros((50, 50)), cmap='gray')
    plt.pause(.001)
    # fig.canvas.draw()
    # plt.show(block=False)

    for _ in range(100):
        state = cam.capture_image()
        plt.imshow(state)
        plt.pause(.001)
        print("done")


def plot_reward_live():
    cam = Camera()

    plt.subplot(221)
    hl, = plt.plot([], [])

    ax1 = plt.gca()

    plt.subplot(222)
    h2, = plt.plot([], [])

    ax2 = plt.gca()

    plt.subplot(223)
    plt.xlabel('action')
    plt.ylabel('q-value')
    # plt.bar
    q_values = [-4, -2, -1, -2, -4, -1]
    bar_plot = plt.bar(list(range(6)), q_values)
    bar_ax = plt.gca()

    plt.subplot(224)
    cam_view = plt.imshow(np.zeros((50, 50)), cmap='gray')
    cam_view.norm.vmax = 1

    for i in range(50):
        x = [.1 * i]
        y1 = [np.sin(x)]
        y2 = [np.cos(x)]

        q_values = [x * 1.5 for x in q_values]

        for rect, q_val in zip(bar_plot, q_values):
            rect.set_height(q_val)

        # bar_ax.set_ylim(np.min(q_values), np.max(q_values))
        bar_ax.relim()
        bar_ax.autoscale_view()

        hl.set_xdata(np.append(hl.get_xdata(), x))
        hl.set_ydata(np.append(hl.get_ydata(), y1))
        h2.set_xdata(np.append(h2.get_xdata(), x))
        h2.set_ydata(np.append(h2.get_ydata(), y2))
        ax1.relim()
        ax1.autoscale_view()
        ax2.relim()
        ax2.autoscale_view()

        cam_view.set_data(cam.capture_image())

        plt.draw()
        plt.pause(.001)

    plt.show()


def save_statistics():
    a = np.array([1, 2, 3, 4, 5])
    b = np.array([11, 12, 13, 14, 15])

    c = np.dstack((a, b))
    c = c.reshape(c.shape[1:])
    print(c.shape)

    # time.strftime("%Y%m%d-%H%M%S")
    np.savetxt("test.csv", c, header=("steps", "reward"))


def updateTargetGraph(tfVars, tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx, var in enumerate(tfVars[0:total_vars // 2]):
        op_holder.append(tfVars[idx + total_vars // 2].assign(
            (var.value() * tau) + ((1 - tau) * tfVars[idx + total_vars // 2].value())))
    return op_holder


def updateTarget(op_holder, sess):
    for op in op_holder:
        sess.run(op)


def train_tf_nn():
    tau = .001

    prediction_network = factory.QNetwork([40, 40], 5)
    target_network = factory.QNetwork([40, 40], 5)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    trainables = tf.trainable_variables()

    targetOps = updateTargetGraph(trainables, tau)

    with tf.Session() as sess:
        sess.run(init)

        for _ in range(100):
            states = np.random.rand(1, 40, 40)

            q_target = np.random.rand(1, 5)
            actions = np.random.randint(6, size=(1))

            sess.run(prediction_network.updateModel,
                     feed_dict={prediction_network.state: states, prediction_network.keep_prob: .5,
                                prediction_network.targetQ: q_target,
                                prediction_network.actions: actions})

        updateTarget(targetOps, sess)

        # print(states)
        states = np.random.rand(1, 40, 40)

        Q1 = prediction_network.Qout.eval(feed_dict={prediction_network.state: states, prediction_network.keep_prob: 1})
        Q2 = target_network.Qout.eval(feed_dict={target_network.state: states, target_network.keep_prob: 1})

        print(Q1)
        print(Q2)

        pred_action = prediction_network.predict.eval(
            feed_dict={prediction_network.state: states, prediction_network.keep_prob: .5})
        print(pred_action)

        tf.train.write_graph(sess.graph, os.path.join(os.getcwd(), 'my-model'), 'train.pbtxt', )

        #saver.save(sess, os.path.join(os.getcwd(), 'my-model'), global_step=0)
        #saver.save(sess, os.path.join(os.getcwd(), 'my-model'), global_step=5)
        #saver.save(sess, os.path.join(os.getcwd(), 'my-model'), global_step=10)


def restore_model():
    sess = tf.Session()
    # First let's load meta graph and restore weights
    saver = tf.train.import_meta_graph('my_test_model-0.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./'))

def button():
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button

    freqs = np.arange(2, 20, 3)

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)
    t = np.arange(0.0, 1.0, 0.001)
    s = np.sin(2 * np.pi * freqs[0] * t)
    l, = plt.plot(t, s, lw=2)

    class Index(object):
        ind = 0

        def next(self, event):
            self.ind += 1
            i = self.ind % len(freqs)
            ydata = np.sin(2 * np.pi * freqs[i] * t)
            l.set_ydata(ydata)
            plt.draw()

        def prev(self, event):
            self.ind -= 1
            i = self.ind % len(freqs)
            ydata = np.sin(2 * np.pi * freqs[i] * t)
            l.set_ydata(ydata)
            plt.draw()

    callback = Index()
    axprev = plt.axes([0.0, 0.05, 0.1, 0.075])
    axnext = plt.axes([0.11, 0.05, 0.1, 0.075])
    bnext = Button(axnext, 'Next')
    bnext.on_clicked(callback.next)
    bprev = Button(axprev, 'Previous')
    bprev.on_clicked(callback.prev)

    plt.show()

if __name__ == "__main__":
    # save_array()
    # load_array()

    # save_model()
    # import csv

    # a = [[np.random.random()*10 for _ in range(r+5)] for r in range(10)]

    # with open("output.csv", "w") as f:
    #    writer = csv.writer(f)
    #    writer.writerows(a)

    # print(np.random.random_integers(0,50,5))

    # read_arduino()

    # stream_webcam()

    # plot_reward_live()

    # save_statistics()

    # train_tf_nn()
    button()