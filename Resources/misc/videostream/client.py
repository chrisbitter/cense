from urllib.request import urlopen
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='Stream Camera Images.')
parser.add_argument('-D', '--dim', type=int, nargs='+', default=[40, 40, 3])
parser.add_argument('-H', '--host', default='0.0.0.0')
parser.add_argument('-P', '--port', default='5000')

args = parser.parse_args()

dim = tuple(args.dim)
url = 'http://' + args.host + ':' + args.port


plt.figure()
plot = plt.imshow(np.zeros(dim), vmin=-1, vmax=1)
plt.ion()
plt.show()
while True:
    plt.pause(.01)

    response = urlopen(url)
    data = response.read()
    img = np.fromstring(data, np.float64).reshape(dim)

    img = (img + 1) / 2

    plot.set_data(img)
    plt.draw()
