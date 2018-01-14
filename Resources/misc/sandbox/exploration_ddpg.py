import matplotlib.pyplot as plt
import numpy as np

# returns 1 if x >= 0, else -1
def sign(x):
    return 2*(x >= 0) - 1

N = 100000

#U = np.random.uniform(0,1,N)

samples = np.zeros(N)

mu = .3

exploration = 0.8

sigma1 = mu*exploration
sigma2 = mu*exploration

rejections = 0

for i in range(N):
    while True:

        #if np.random.random()


        if np.random.random() < abs(mu):
            # sample value for interval [0, -mu] or [0, mu]
            samples[i] = np.random.normal(0, abs(mu*exploration))

            if np.random.random() >= exploration:
                # value in interval [0, mu]
                samples[i] = mu - sign(mu) * abs(samples[i])
            else:
                # value in interval [0, -mu]
                samples[i] = -mu + sign(mu) * abs(samples[i])
        else:
            # sample value for interval [-mu, -sign(mu)] or [mu, sign(mu)]
            samples[i] = np.random.normal(0, abs((1-abs(mu)) * exploration))

            if np.random.random() >= exploration:
                # value in interval [mu, sign(mu)]
                samples[i] = mu + sign(mu)*abs(samples[i])
            else:
                # value in interval [0, -mu]
                samples[i] = -mu - sign(mu)*abs(samples[i])

        if -1 <= samples[i] <= 1:
            break
        else:
            rejections += 1

print(rejections)
print(rejections/N)

plt.hist(samples, bins=500, range=[-1,1])

plt.plot()
plt.show()
