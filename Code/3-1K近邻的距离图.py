import math
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
r = 1

linestyle = ['b-','k-','m-','r-','y-']
p_values = (0.25, 0.5, 1, 2, 4,100)

for i,p in enumerate(p_values):
    x = np.arange(-r,r+1e-5,1/128.0)
    y = (r**p - (abs(x)**p))**(1.0/p)
    plt.plot(x,y,x,-y)

ax = plt.gca()
ax.set_aspect(1)
plt.show()

def L(x, y, p=2):
    # x1 = [1, 1], x2 = [5,1]
    if len(x) == len(y) and len(x) > 1:
        sum = 0
        for i in range(len(x)):
            sum += math.pow(abs(x[i] - y[i]), p)
        return math.pow(sum, 1/p)
    else:
        return 0

x1 = [1, 1]
x2 = [5, 1]
x3 = [4, 4]

def L(x, y, p=2):
    # x1 = [1, 1], x2 = [5,1]
    if len(x) == len(y) and len(x) > 1:
        sum = 0
        for i in range(len(x)):
            sum += math.pow(abs(x[i] - y[i]), p)
        return math.pow(sum, 1/p)
    else:
        return 0

