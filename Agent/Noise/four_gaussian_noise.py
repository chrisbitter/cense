import matplotlib.pyplot as plt
import numpy as np


class FGN(object):
    # returns 1 if x >= 0, else -1
    def sign(self, x):
        if x > 0:
            return 1
        elif x < 0:
            return -1
        else:
            # random -1 or 1
            return 2 * (np.random.random() >= 0.5) - 1

    def noise(self, x, exploration):

        while True:

            if np.random.random() < abs(x):
                # sample value for interval [0, -mu] or [0, mu]
                noise = np.random.normal(0, abs(x * exploration))

                if np.random.random() >= exploration:
                    # value in interval [0, mu]
                    noise = x - self.sign(x) * abs(noise)
                else:
                    # value in interval [0, -mu]
                    noise = -x + self.sign(x) * abs(noise)
            else:
                # sample value for interval [-mu, -sign(mu)] or [mu, sign(mu)]
                noise = np.random.normal(0, abs((1 - abs(x)) * exploration))

                if np.random.random() >= exploration:
                    # value in interval [x, self.sign(x)]
                    noise = x + self.sign(x) * abs(noise)
                else:
                    # value in interval [0, -x]
                    noise = -x - self.sign(x) * abs(noise)

            if -1 <= noise <= 1:
                break

        return noise

    def noise_2(self, x, exploration, one_sided=False):

        while True:

            if np.random.random() < abs(x):
                # sample value for interval [0, -mu] or [0, mu]
                noise = np.random.normal(0, abs(x * exploration))

                if np.random.random() >= exploration or one_sided:
                    # value in interval [0, mu]
                    noise = x - self.sign(x) * abs(noise)
                else:
                    # value in interval [0, -mu]
                    noise = -x + self.sign(x) * abs(noise)
            else:
                # sample value for interval [-mu, -sign(mu)] or [mu, sign(mu)]
                noise = np.random.normal(0, abs((1 - abs(x)) * exploration))

                if np.random.random() >= exploration or one_sided:
                    # value in interval [x, self.sign(x)]
                    noise = x + self.sign(x) * abs(noise)
                else:
                    # value in interval [0, -x]
                    noise = -x - self.sign(x) * abs(noise)

            if -1 <= noise <= 1:
                break

        return noise


if __name__ == '__main__':

    N = 100000

    samples = np.zeros(N)

    action = .7

    fgn = FGN()

    exploration = .3

    for i in range(N):
        # samples[i] = fgn.noise_2(action, exploration, False)
        samples[i] = fgn.noise(action, exploration)

    plt.hist(samples, bins=500, range=[-1, 1], normed=True)

    plt.plot()
    plt.show()
