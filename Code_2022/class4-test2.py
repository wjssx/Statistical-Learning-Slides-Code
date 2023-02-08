import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

r = 1

linestyle = ['b-', 'k-', 'm-', 'r-', 'y-']
p_values = (0.25, 0.5, 1, 2, 4, 100)

for i, p in enumerate(p_values):
    x = np.arange(-r, r + 1e-5, 1 / 128.0)
    y = (r ** p - (abs(x) ** p)) ** (1.0 / p)
    plt.plot(x, y, x, -y)

ax = plt.gca()
ax.set_aspect(1)
plt.show()

#####
X = [[0], [1], [2], [3]]
y = [0, 0, 1, 1]
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X, y)

print(neigh.predict([[1.1]]))
print(neigh.predict_proba([[0.9]]))
