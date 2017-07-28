import matplotlib.pyplot as plt
import numpy as np

N = 100000

#U = np.random.uniform(0,1,N)

samples = np.zeros(N)

mu = .5

exploration = 0

sigma1 = mu*exploration
sigma2 = mu*exploration

rejections = 0

for i in range(N):
    while True:
        #if np.random.uniform() < exploration:
        if False:
            samples[i] = np.random.normal(-mu, sigma2)
        else:
            samples[i] = np.random.normal(mu, sigma1)
        if -1 <= samples[i] <= 1:
            break
        else:
            rejections += 1

print(rejections)
print(rejections/N)

plt.hist(samples, bins=100)

plt.plot()
plt.show()
