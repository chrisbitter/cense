import matplotlib.pyplot as plt
import numpy as np

def emerging_gaussian(x, exploration, low=-1, high=1):
    if not low <= x <= high:
        print('Wrong parameters: Must be low <= x <= high: %f, %f, %f' % (low, x, high))

        if x <= low:
            x = low
            print('clipping x to low')
        else:
            x = high
            print('clipping x to high')

    if not 0 <= exploration <= 1:
        raise IOError('Wrong parameters: Must be 0 <= exploration <= 1: %f' % exploration)

    if np.random.random() >= exploration:
        if np.random.random() <= (x-low)/(high-low):
            # left side
            std = abs((x-low) * exploration)
            while True:
                sample = abs(np.random.normal(0, std))
                if low <= x - abs(sample) <= high:
                    return x - abs(sample)

        else:
            # right side
            std = abs((high-x) * exploration)
            while True:
                sample = abs(np.random.normal(0, std))
                if low <= x + abs(sample) <= high:
                    return x + abs(sample)
    else:
        return np.random.uniform(low, high)

if __name__ == '__main__':

    N = 100000

    samples = np.zeros(N)

    action = 1

    exploration = .3

    for i in range(N):
        # samples[i] = fgn.noise_2(action, exploration, False)
        samples[i] = emerging_gaussian(action, exploration, -1, 1)

    plt.hist(samples, bins=500, range=[-1, 1], normed=True)

    plt.plot()
    plt.show()
