from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

pca = PCA(2)
print(pca)

data = load_iris()
X, y = data.data, data.target
X_proj = pca.fit_transform(X)
print(X_proj.shape)

plt.scatter(X_proj[:, 0], X_proj[:, 1], c=y)
plt.show()
