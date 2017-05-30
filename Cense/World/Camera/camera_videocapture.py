import pygame
import pygame.camera
import numpy as np
from scipy.misc import imresize
import time
import matplotlib.pyplot as plt
import matplotlib.image

from VideoCapture import Device


class Camera(object):
    __camera = None
    SIZE = None

    def __init__(self, size=(50, 50)):

        self.__camera = Device(1)

        self.SIZE = size

    def shutdown(self):
        del self.__camera
        pygame.quit()

    def capture_image(self):
        # take picture
        frame = self.__camera.getImage()

        # convert to array
        rgb = np.array(frame.getdata(),
                         np.uint8).reshape(frame.size[1], frame.size[0], 3)

        rgb_cropped = rgb[180:,200:500,:]

        # convert to gray image
        gray = np.dot(rgb_cropped[..., :3], [.299, .587, .114])

        #gray = gray[200:][200:]
        # rescale image
        # state = gray
        state = imresize(gray, self.SIZE)

        return state


    def calibrate_camera(self):
        # take picture
        frame = self.__camera.getImage()

        plt.figure()

        # convert to array
        rgb = np.array(frame.getdata(),
                         np.uint8).reshape(frame.size[1], frame.size[0], 3)

        plt.subplot(221)
        plt.imshow(rgb)

        print(rgb.shape)

        rgb_cropped = rgb[180:,200:500,:]

        print(rgb_cropped.shape)

        plt.subplot(222)
        plt.imshow(rgb_cropped)

        # convert to gray image
        gray = np.dot(rgb_cropped[..., :3], [.299, .587, .114])

        plt.subplot(223)
        plt.imshow(gray, cmap='gray')

        #gray = gray[200:][200:]
        # rescale image
        # state = gray
        state = imresize(gray, self.SIZE)

        plt.subplot(224)
        plt.imshow(state, cmap='gray')

        plt.show()

        return state

def take_picture():
    cam = Camera((50, 50))
    image = cam.capture_image()
    plt.figure()
    plt.imshow(image, cmap='gray')

    plt.show()
    cam.shutdown()


def test_1():
    timeout_vals = []
    max_vals = []

    cam = Camera()

    for timeout in range(0, 51, 10):
        max_val = 0

        print("timeout: %is" % timeout)
        start = time.time()
        while time.time() - start < timeout:
            pass

        # take 50 pictures
        for p in range(50):
            start = time.time()
            cam.capture_image(False)
            end = time.time()
            duration = end - start

            if duration > max_val:
                max_val = duration

        timeout_vals.append(timeout)
        max_vals.append(max_val)
        print(max_val)
    plt.scatter(timeout_vals, max_vals)
    plt.show()
    cam.shutdown()


if __name__ == '__main__':
    cam = Camera((50, 50))
    cam.calibrate_camera()
    cam.shutdown()