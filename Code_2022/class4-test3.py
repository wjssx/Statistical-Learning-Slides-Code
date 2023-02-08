import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

irisData = datasets.load_iris()

X = irisData.data[:, :4]
y = irisData.target

weights = 'uniform'
n_neighbors=15
# we create an instance of Neighbours Classifier and fit the data.
classifier = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
classifier.fit(X, y)

print('KNN classifier accuracy - "%s" - %.3f' % (weights, classifier.score(X, y)))
