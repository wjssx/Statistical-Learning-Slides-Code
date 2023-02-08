import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

digits = load_digits()
print(digits.keys())

# looking at data, there looks to be 64 features, what are these?
print(digits.data.shape)
# another available dataset is called images. Let's check this out.
print(digits.images.shape)

import matplotlib.pyplot as plt

plt.gray()
plt.matshow(digits.images[0])
plt.show()

X, y = digits.data, digits.target
pca_digits = PCA(0.95)
X_proj = pca_digits.fit_transform(X)
print(X.shape, X_proj.shape)

# Let's run PCA with 2 components so as to plot the data in 2D
pca_digits = PCA(2)
X_proj = pca_digits.fit_transform(X)
print(np.sum(pca_digits.explained_variance_ratio_))
# Note we only retain about 28% of the variance by choosing 2 components

print(X_proj.shape)

# Let's plot the principal components as a scatter plot
plt.scatter(X_proj[:, 0], X_proj[:, 1], c=y)
plt.colorbar()
plt.show()

pca_digits = PCA(64).fit(X)
plt.semilogx(np.cumsum(pca_digits.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance retained')
plt.ylim(0, 1)
plt.show()
