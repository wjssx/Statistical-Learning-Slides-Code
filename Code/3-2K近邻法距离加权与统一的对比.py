

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets


# import some data to play with
irisData = datasets.load_iris()
irisData.data[0:5 ,:]


X = irisData.data[:, :2]
y = irisData.target
X[:10 ,:]


n_neighbors = 15


step = .01  # step size in the mesh

for weights in ['uniform', 'distance']:
    # we create an instance of Neighbours Classifier and fit the data.
    classifier = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    classifier.fit(X, y)

    print('KNN classifier accuracy - "%s" - %.3f' % (weights ,classifier.score(X ,y)))

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    x_grid, y_grid = np.meshgrid(np.arange(x_min, x_max, step = step),
                                 np.arange(y_min, y_max, step = step))
    Z = classifier.predict(np.c_[x_grid.ravel(), y_grid.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(x_grid.shape)
    plt.figure()
    plt.pcolormesh(x_grid, y_grid, Z, cmap=ListedColormap(['lightblue', 'lightgreen', 'lightyellow']) )

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y,
                edgecolor='k', s=20)
    plt.xlim(x_grid.min(), x_grid.max())
    plt.ylim(y_grid.min(), y_grid.max())
    plt.title("KNN 3-Class Classification (k = %d, weights = '%s')"
              % (n_neighbors, weights))


plt.show()

