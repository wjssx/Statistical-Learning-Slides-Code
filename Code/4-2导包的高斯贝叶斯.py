import numpy as np

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
Y = np.array([1, 1, 1, 2, 2, 2])

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB(priors=None, var_smoothing=1e-09)
clf.fit(X, Y)
print(clf.predict([[-0.8, -1]]))