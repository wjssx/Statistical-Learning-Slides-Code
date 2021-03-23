import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import random as sparse_random
from sklearn.random_projection import sparse_random_matrix
X = sparse_random(100, 100, density=0.01, format='csr',
                   random_state=42)
svd = TruncatedSVD(n_components=5, n_iter=7, random_state=42)
svd.fit(X)

print(svd.explained_variance_ratio_)

print(svd.explained_variance_ratio_.sum())
print(svd.singular_values_)
import numpy as np

a = np.random.randn(9, 6) + 1j*np.random.randn(9, 6)
b = np.random.randn(2, 7, 8, 3) + 1j*np.random.randn(2, 7, 8, 3)

u, s, vh = np.linalg.svd(a, full_matrices=True)
print(a.shape)
print(u.shape, s.shape, vh.shape)

from PIL import Image
import matplotlib.image as mpimg


I = mpimg.imread('data/F_test.jpeg')
#Now, let's look at the size of this numpy array object img as well as plot it using imshow.
print(I.shape)
plt.axis('off')
plt.imshow(I)

def show_img(img):
    plt.figure(figsize = (10, 7.5))
    plt.imshow(img, cmap = 'gray', vmin=0, vmax=255, aspect = 'auto')
    plt.axis('off')
    plt.show()

U, S, V_T = np.linalg.svd(I)
#U.shape, S.shape, V_T.shape



I = I[:,:,1]
print(I.shape)


plt.figure(figsize = (9, 5))
plt.plot(np.arange(S.shape[0]), S)
plt.yscale('log')
plt.xlabel('Index of $\sigma$')
plt.ylabel('log(value of $\sigma$)')
plt.title('Singular values $\sigma_i$ vs its index')
plt.show()
plt.figure(figsize = (9, 5))
plt.plot(np.cumsum(S) / sum(S))
plt.xlabel('Index of $\sigma$')
plt.ylabel('Value of $\sigma$')
plt.title('Cumulative sum of $\sigma_i$ vs its index\n(Percent of explained variance)')
plt.show()
S_full = np.zeros((U.shape[0], V_T.shape[0]))

#S_full.shape

S_diag = np.diag(S)
S_full[:S_diag.shape[0], :S_diag.shape[1]] = S_diag

for i in [5, 10, 25, 50, 100, 200, U.shape[0]]:
    print(str(i) + '\n')
    show_img(U[:, :i].dot(S_full[:i, :i].dot(V_T[:i, :])))
    print('-' * 100 + '\n')


