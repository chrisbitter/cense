import numpy as np

a = np.array(range(361))
b = np.array(range(361))

c = a - b
d = c + 180
e = d % 360
f = e - 180
g = np.absolute(f)
h = np.max(g)

print(c,d,e,f,g,h, sep="\n")