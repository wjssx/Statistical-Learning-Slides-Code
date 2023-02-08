import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

import math

p = np.linspace(0.01, 1, num=50, endpoint=False)

entropy = -p * np.log2(p) - (1 - p) * np.log2(1 - p)

# plt.plot(b)
plt.plot(p, entropy)
plt.grid(True)
plt.xlabel('p')
plt.ylabel('Entropy(bit)')
# plt.plot(p,gini)

max_en = 2 * (-(1 / 2) * np.log2(1 / 2))
print(max_en)

d = np.linspace(0.01, 100, num=50, endpoint=False)
ld = np.log2(d)
plt.show()
