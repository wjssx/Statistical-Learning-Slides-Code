import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def pca(data, n_dim):
    '''
    pca is O(D^3)
    :param data: (n_samples, n_features(D))
    :param n_dim: target dimensions
    :return: (n_samples, n_dim)
    '''
    data = data - np.mean(data, axis=0, keepdims=True)

    cov = np.dot(data.T, data)

    eig_values, eig_vector = np.linalg.eig(cov)
    # print(eig_values)
    indexs_ = np.argsort(-eig_values)[:n_dim]
    picked_eig_values = eig_values[indexs_]
    picked_eig_vector = eig_vector[:, indexs_]
    data_ndim = np.dot(data, picked_eig_vector)
    return data_ndim


data = load_iris()
X = data.data
Y = data.target
data_2d1 = pca(X, 2)
plt.figure(figsize=(8, 4))
plt.subplot(121)
plt.title("my_PCA")
plt.scatter(data_2d1[:, 0], data_2d1[:, 1], c=Y)
plt.show()