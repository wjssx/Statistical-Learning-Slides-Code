# SparsePCA
import numpy as np
from sklearn.datasets import make_friedman1, load_digits
from sklearn.decomposition import SparsePCA

X, _ = load_digits(return_X_y=True)
transformer = SparsePCA(n_components=5, random_state=0)
transformer.fit(X)
X_transformed = transformer.transform(X)

print(X_transformed.shape)

# KernelPCA
from sklearn.datasets import load_digits
from sklearn.decomposition import KernelPCA

X, y = load_digits(return_X_y=True)
print(X.shape)
transformer = KernelPCA(n_components=7, kernel='linear')
X_transformed = transformer.fit_transform(X)
print(X_transformed.shape)

# Isomap
from sklearn.manifold import Isomap

isomap = Isomap(n_components=2, n_neighbors=5)
new_X_isomap = isomap.fit_transform(X)
print(new_X_isomap.shape)

from sklearn.manifold import TSNE

X_embedded = TSNE(n_components=2).fit_transform(X)
print(X_embedded.shape)
