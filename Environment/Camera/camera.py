from VideoCapture import Device
from threading import Lock, Thread
from multiprocessing import Queue
import numpy as np
from scipy.misc import imresize


class Camera(object):

    def __init__(self, size):

        print("Setup Camera")

        self.run = True
        self.camera = Device(0)
        self.SIZE = size
        self.decision_lock = Lock()
        self.read_lock = Lock()
        self.frames = [self.take_picture()] * 2
        self.write_index = 0
        self.thread = Thread(target=self.record)
        self.thread.start()

    def shutdown(self):
        self.run = False

    def record(self):
        while self.run:
            if self.read_lock.acquire(False):
                self.write_index += 1
                self.write_index %= 2
                self.read_lock.release()

            self.frames[self.write_index] = self.take_picture()

    def get_frame(self):
        # take picture
        with self.read_lock:

            read_index = (self.write_index + 1) % 2

            return self.frames[read_index]

    def take_picture(self):
        frame = self.camera.getImage()

        # convert to array
        rgb = np.array(frame.getdata(), np.uint8).reshape(frame.size[1], frame.size[0], 3)
        result_rgb = imresize(rgb, self.SIZE) / 255

        return result_rgb

if __name__ == '__main__':

    import matplotlib.pyplot as plt

    dim = (360, 360, 3)

    cam = Camera(dim)

    import time

    plt.ion()
    plt.show()
    plot = plt.imshow(np.zeros(dim), vmin=-1, vmax=1)

    while True:
        time.sleep(1)
        frame = cam.get_frame()
        plot.set_data(frame)
        plt.draw()
        plt.pause(.0001)

    cam.shutdown()