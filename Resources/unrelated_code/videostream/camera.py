import pygame.camera
import numpy as np
from scipy.misc import imresize
import matplotlib.pyplot as plt

class Camera(object):
    def __init__(self, dim=(40,40,3)):

        self.dim = dim

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

        return img.tostring()

        # return Image.fromarray(rgb, 'RGB')