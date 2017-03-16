import numpy as np
import matplotlib.image as plt
from PIL import Image

# world resolution
world_x = 25
world_y = 15


def create_wire_from_file(file_path='Resources/wires/HotWire2.png', debug=False):
    img = Image.open(file_path)
    # resize image in-place
    img.thumbnail((world_x, world_y), Image.ANTIALIAS)
    img_arr = np.asarray(img)

    img_arr_cp = np.copy(img_arr)
    img_arr_cp[img_arr_cp > 20] = 255
    img_arr_cp[img_arr_cp <= 20] = 0

    # Visualize the array containing the world
    if debug:
        plt.imshow(img_arr_cp, interpolation='none')
        plt.show()

    wire_arr = np.array(img_arr_cp[:, :, 0])
    wire_arr[wire_arr > 0] = 1.0

    print(wire_arr.shape)

    return wire_arr
