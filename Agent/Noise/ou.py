import matplotlib.pyplot as plt
import numpy as np


class OU(object):

    def noise(self, x, mu, theta, sigma):

        for i in range(x.shape[0]):
            x[i] += theta*(mu[i] - x[i])
            x[i] += sigma * (2 * np.random.random() - 1)

        return x


if __name__ == '__main__':

    N = 100
    trials = 5

    action = np.array([0])

    samples = np.empty(shape=(trials, N) + action.shape)
 #       (trials, N, action.shape))


    ou = OU()

    for t in range(trials):

        action = np.array([0.])

        for i in range(N):
            samples[t, i] = action
            action = ou.noise(action, 0, .2, 0.4)

    print(samples.shape)

    for t in range(trials):
        plt.plot(samples[t])

    #plt.hist(samples, bins=500, range=[-1, 1], normed=True)

    plt.show()
