import numpy as np
from PIL import Image
import matplotlib.image as mpimg

# import cv2
import matplotlib.pyplot as plt

# %matplotlib inline

I = mpimg.imread('data/header.jpg')
# Now, let's look at the size of this numpy array object img as well as plot it using imshow.
print(I.shape)
plt.axis('off')
plt.imshow(I)
plt.show()


def show_img(img):
    plt.figure(figsize=(10, 7.5))
    plt.imshow(img, cmap='gray', vmin=0, vmax=255, aspect='auto')
    plt.axis('off')
    plt.show()


gray_img = I[:, :, 1]
show_img(gray_img)
print(gray_img.shape)
U, S, V_T = np.linalg.svd(gray_img)
print(U.shape, S.shape, V_T.shape)

# Plot sigmas
plt.figure(figsize=(9, 5))
plt.plot(np.arange(S.shape[0]), S)
plt.yscale('log')
plt.xlabel('Index of $\sigma$')
plt.ylabel('log(value of $\sigma$)')
plt.title('Singular values $\sigma_i$ vs its index')
plt.show()

# Plot cumsum of sigma
plt.figure(figsize=(9, 5))
plt.plot(np.cumsum(S) / np.sum(S))
plt.xlabel('Index of $\sigma$')
plt.ylabel('Value of $\sigma$')
plt.title('Cumulative sum of $\sigma_i$ vs its index\n(Percent of explained variance)')
plt.show()

# Create an empty matrix to fill with sigma values (np.lialg.svd returns sigma as an array)
S_full = np.zeros((U.shape[0], V_T.shape[0]))
print(S_full.shape)

# Populate sigma matrix
S_diag = np.diag(S)
print(S_diag.shape)
S_full[:S_diag.shape[0], :S_diag.shape[1]] = S_diag

# for i in [5, 10, 25, 50, 100, 200, U.shape[0]]:
#     print(str(i) + '\n')
#     show_img(U[:, :i].dot(S_full[:i, :i].dot(V_T[:i, :])))
#     print('-' * 100 + '\n')
#
# print(U[:, :5])
i = 200
print(U[:, :i].shape, S_full[:i, :i].shape, V_T[:i, :].shape)
