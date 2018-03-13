from VideoCapture import Device

import numpy as np
from scipy.misc import imresize


class Camera(object):
    __camera = None
    SIZE = None

    def __init__(self, size):

        print("Setup Camera")

        self.__camera = Device(1)
        self.SIZE = size

    def get_frame(self):
        # take picture
        frame = self.__camera.getImage()

        # convert to array
        rgb = np.array(frame.getdata(), np.uint8).reshape(frame.size[1], frame.size[0], 3)

        result_rgb = (imresize(rgb, self.SIZE) / 127.5 - 1)

        return result_rgb


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    dim = (40, 40, 3)

    cam = Camera(dim)


    plt.ion()
    plt.show()
    plt.figure()
    plot = plt.imshow(np.zeros(dim), vmin=-1, vmax=1)

    import time

    while True:

        t = time.time()


        img = cam.get_frame()

        img = (img + 1) / 2

        plot.set_data(img)
        plt.draw()
        plt.pause(.0001)