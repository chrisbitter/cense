import pygame.camera
import pygame
import numpy as np
from scipy.misc import imresize
import matplotlib.pyplot as plt


class Camera(object):
    def __init__(self, dim=(40, 40, 3)):
        self.dim = dim

        pygame.init()
        pygame.camera.init()

        cameras = pygame.camera.list_cameras()

        self.camera = pygame.camera.Camera(cameras[0])

        self.camera.start()

        self.camera.get_image()

    def __del__(self):
        self.camera.stop()

    def get_frame(self):
        frame = self.camera.get_image()

        img = np.array(pygame.surfarray.array3d(frame))

        img = imresize(img, self.dim)

        img = img / 127.5 - 1

        return img

        # return Image.fromarray(rgb, 'RGB')


if __name__ == '__main__':

    dim = (640, 480, 3)

    cam = Camera(dim)

    plt.ion()
    plt.show()
    plt.figure()
    plot = plt.imshow(np.zeros(dim), vmin=-1, vmax=1)

    import time

    while True:

        t = time.time()
        time.sleep(4)

        img = cam.get_frame()

        img = (img + 1) / 2

        plot.set_data(img)
        plt.draw()
        plt.pause(.0001)