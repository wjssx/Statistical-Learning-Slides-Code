import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import numpy as np
iris = load_iris()
#checking to see what datasets are available in iris
print(iris.keys())
print(iris.data.shape)
print(iris.feature_names)

from sklearn.decomposition import PCA
pca = PCA(2)
print(pca)

X, y = iris.data, iris.target
X_proj = pca.fit_transform(X)
print(X_proj.shape)

plt.scatter(X_proj[:,0], X_proj[:,1],c=y)
plt.show()

from sklearn.datasets import load_digits
digits = load_digits()
print(digits.keys())

print(digits.data.shape)

print(digits.images.shape)

X,y = digits.data, digits.target
pca_digits=PCA(0.95)
X_proj = pca_digits.fit_transform(X)
print(X.shape, X_proj.shape)


pca_digits=PCA(2)
X_proj = pca_digits.fit_transform(X)
print(np.sum(pca_digits.explained_variance_ratio_))


print(X_proj.shape)


plt.scatter(X_proj[:,0], X_proj[:,1], c=y)
plt.colorbar()
plt.show()

pca_digits = PCA(64).fit(X)
plt.semilogx(np.cumsum(pca_digits.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance retained')
plt.ylim(0,1)
plt.show()

from PIL import Image

im1 = Image.open('data/F_test.jpeg')
im1.save('data/F_test.png')

import matplotlib.image as mpimg
img = mpimg.imread('data/F_test.png')
#Now, let's look at the size of this numpy array object img as well as plot it using imshow.
print(img.shape)
plt.axis('off')
plt.imshow(img)


img_r = np.reshape(img, (800, 3600))
print(img_r.shape)


ipca = PCA(64).fit(img_r)
img_c = ipca.transform(img_r)
print(img_c.shape)
print(np.sum(ipca.explained_variance_ratio_))


temp = ipca.inverse_transform(img_c)
print(temp.shape)


temp = np.reshape(temp, (800,1200,3))
print(temp.shape)


plt.axis('off')
plt.imshow(temp)
