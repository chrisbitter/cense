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

    def capture_image(self):
        # take picture
        frame = self.__camera.getImage()

        # convert to array
        rgb = np.array(frame.getdata(),
                       np.uint8).reshape(frame.size[1], frame.size[0], 3)

        # rgb_cropped = rgb[100:, 120:500, :]
        #
        # # print(rgb_cropped.shape)
        #
        #
        # # rgb_cropped = rgb
        #
        # gray = np.dot(rgb[..., :3], [.299, .587, .114])
        #
        # gray_cropped = np.dot(rgb[100:, 120:500, :][..., :3], [.299, .587, .114])
        #
        # result_original = rgb

        result_rgb = (imresize(rgb, self.SIZE) / 255)

        # result_gray = (imresize(gray, self.SIZE) / 255)

        # result_gray_cropped = (imresize(gray_cropped, self.SIZE) / 255)

        return result_rgb

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    cam = Camera((40, 40, 3))

    orig, rgb, gray, gray_c = cam.capture_image()

    plt.figure()
    plt.imshow(orig)
    plt.figure()
    plt.imshow(rgb)
    plt.figure()
    plt.imshow(gray, cmap='gray')
    plt.figure()
    plt.imshow(gray_c, cmap='gray')

    plt.show()
