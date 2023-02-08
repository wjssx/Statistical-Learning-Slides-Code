import numpy as np
import matplotlib.pyplot as plt


# %matplotlib inline

def sigmod(x):
    return 1 / (1 + np.exp(-x))


x = np.arange(-10, 10., 0.1)
y = sigmod(x)

plt.plot(x, y)
plt.grid(True)
plt.show()
