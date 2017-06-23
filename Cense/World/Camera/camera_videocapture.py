from VideoCapture import Device

import numpy as np
from scipy.misc import imresize
import matplotlib.pyplot as plt


class Camera(object):
    __camera = None
    SIZE = None

    def __init__(self, size, set_status_func):

        self.set_status_func = set_status_func
        self.set_status_func("Setup Camera")

        self.__camera = Device(0)
        self.SIZE = size

    def capture_image(self):
        # take picture
        frame = self.__camera.getImage()

        # convert to array
        rgb = np.array(frame.getdata(),
                       np.uint8).reshape(frame.size[1], frame.size[0], 3)

        rgb_cropped = rgb[100:, 130:500, :]

        gray = np.dot(rgb_cropped[..., :3], [.299, .587, .114])

        return imresize(gray, self.SIZE) / 255

if __name__ == '__main__':
    cam = Camera((40, 40), print)

    state = cam.capture_image()
    plt.imshow(state, cmap='gray')
    plt.show()
