import matplotlib.image as mpimg
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

img = mpimg.imread('data/header.jpg')
# Now, let's look at the size of this numpy array object img as well as plot it using imshow.
print(img.shape)
plt.axis('off')
plt.imshow(img)
plt.show()

img_r = np.reshape(img, (800, 3600))
print(img_r.shape)

ipca = PCA(64).fit(img_r)
img_c = ipca.transform(img_r)
print(img_c.shape)
print(np.sum(ipca.explained_variance_ratio_))

print(ipca)

# OK, now to visualize how PCA has performed this compression, let's inverse transform the PCA output and
# reshape for visualization using imshow.
temp = ipca.inverse_transform(img_c)
print(temp.shape)
# reshaping 2988 back to the original 996 * 3
temp = np.reshape(temp, (800, 1200, 3))
print(temp.shape)

plt.axis('off')
plt.imshow(temp)
plt.imshow(temp.astype('uint8'))
plt.show()
