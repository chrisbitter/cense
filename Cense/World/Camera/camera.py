import pygame
import pygame.camera
import numpy as np
from scipy.misc import imresize
import time
import matplotlib.pyplot as plt
import matplotlib.image


class Camera(object):
    __camera = None
    SIZE = None

    def __init__(self, size=(50, 50)):
        # init camera
        pygame.camera.init()
        cameras = pygame.camera.list_cameras()
        self.__camera = pygame.camera.Camera(cameras[0])
        self.__camera.start()

        # take one image to get camera ready
        self.__camera.get_image()
        self.SIZE = size

    def shutdown(self):
        self.__camera.stop()
        pygame.quit()

    def capture_image(self, debug, imgpath=''):
        # take picture
        frame = self.__camera.get_image()

        # convert to array
        rgb = np.array(pygame.surfarray.pixels3d(frame))

        # print(rgb.shape)

        if debug:
            rgb = matplotlib.image.imread(imgpath)

        # convert to gray image
        gray = np.dot(rgb[..., :3], [.299, .587, .114])

        # rescale image
        state = imresize(gray, self.SIZE)

        return state


def debug_with_pictures():
    cam = Camera()
    for i in range(1, 4):
        path = "Draht_" + str(i) + ".png"
        start = time.time()
        image = cam.capture_image(True, path)
        end = time.time()
        print(end - start)

        plt.figure()
        plt.imshow(image, cmap='gray')
    plt.plot()
    cam.shutdown()

def take_picture():
    cam = Camera((50, 50))
    image = cam.capture_image(False)
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
    # debug_with_pictures()
    take_picture()
    # test_1()
