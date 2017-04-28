import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys

# world resolution
world_resolution = 25, 15


#
# Takes an image and creates a three dimensional array from it
# First and second dimensions being the coordinates of the pixel
# Third dimension being either 1 or 0 where a 1 represents the wire and a 0 represents the background
#
def create_wire_from_file(file_path=sys.path[2] + '/../Resources/wires/Cense_wire_01.png', debug=False):

    # Read image and export it as numpy array
    image = Image.open(file_path)
    np_image = np.asarray(image, dtype=np.uint8)
    img_arr_cp = np.copy(np_image)

    # Visualize the array containing the world
    if debug:
        plt.imshow(img_arr_cp)
        plt.show()

    # Reduce the array to the red values
    wire_arr = np.array(img_arr_cp[:, :, 0])
    # Where red values aren't 0 aka where the wire is put a 1
    wire_arr[wire_arr > 0] = 1.0

    # Rotate the image array, so it can be accessed naturally with coordinates like self.__world[x,y]
    wire_arr_temp = np.zeros_like(wire_arr)
    # Turn around all lines
    for i in range(wire_arr.shape[0]):
        wire_arr_temp[i] = wire_arr[i][::-1]
    # wire_arr = np.fliplr(np.flipud(wire_arr_temp.T))
    wire_arr = np.fliplr(wire_arr_temp.T)
    if debug:
        plt.imshow(wire_arr)
        plt.show()

    return wire_arr

if __name__ == '__main__':
    create_wire_from_file(debug=True)
