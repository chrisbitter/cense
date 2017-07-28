import matplotlib.pyplot as plt
import numpy as np

N = 100000

U = np.random.uniform(0,1,N)

samples = np.zeros(N)

mu = .5

exploration = .1

sigma1 = mu*exploration + .1
sigma2 = mu*exploration + .1

for i in range(N):
    if U[i] < .7:
        samples[i] = np.random.normal(mu, sigma1)
    else:
        samples[i] = np.random.normal(-mu, sigma2)

plt.hist(samples, bins=100)

plt.plot()
plt.show()
