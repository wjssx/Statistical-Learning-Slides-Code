
import itertools

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn import mixture

color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold',
                              'darkorange'])


def plot_results(X, Y_, means, covariances, index, title):
    splot = plt.subplot(2, 1, 1 + index)
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.xlim(-9., 5.)
    plt.ylim(-3., 6.)
    plt.xticks(())
    plt.yticks(())
    plt.title(title)


# Number of samples per component
n_samples = 500

# Generate random sample, two components
np.random.seed(0)
C = np.array([[0., -0.1], [1.7, .4]])
X = np.r_[np.dot(np.random.randn(n_samples, 2), C),
          .7 * np.random.randn(n_samples, 2) + np.array([-6, 3])]

# Fit a Gaussian mixture with EM using five components
gmm = mixture.GaussianMixture(n_components=5, covariance_type='full').fit(X)
plot_results(X, gmm.predict(X), gmm.means_, gmm.covariances_, 0,
             'Gaussian Mixture')



plt.show()

print(gmm.weights_)
print(gmm.means_)
print(gmm.covariances_)

from sklearn.cluster import KMeans
import numpy as np
X = np.array([[1, 2], [1, 4], [1, 0],[4, 2], [4, 4], [4, 0]])
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
print(kmeans.labels_)
print(kmeans.predict([[0, 0], [4, 4]]))
print(kmeans.cluster_centers_)


#euclidian distance between 2 data points. For as many data points as necessary.
def euclidean_distance(a, b):
    return np.linalg.norm(a-b)





def kmeans(data,k=3):
    m = data.shape[0]
    index = random.sample(range(m),k)
    mu = data[index] #随机选择初始均值向量


    while True:

        C = defaultdict(list)

        for j in range(0,m):
            dij = [euclidean_distance(data[j],mu[i]) for i in range(k)]
            lambda_j = np.argmin(dij)   #选择最小的值得下标

            C[lambda_j].append(data[j].tolist())

        new_mu = [np.mean(C[i],axis=0).tolist() for i in range(k)]

        if (euclidean_distance(np.array(new_mu),np.array(mu))>1e-9):
            mu = new_mu
        else:
            break

    return C,mu


watermelon = np.array([[ 0.697  ,0.46 ],
                         [ 0.774  ,0.376],
                         [ 0.634  ,0.264],
                         [ 0.608  ,0.318],
                         [ 0.556  ,0.215],
                         [ 0.403  ,0.237],
                         [ 0.481  ,0.149],
                         [ 0.437  ,0.211],
                         [ 0.666  ,0.091],
                         [ 0.243  ,0.267],
                         [ 0.245  ,0.057],
                         [ 0.343  ,0.099],
                         [ 0.639  ,0.161],
                         [ 0.657  ,0.198],
                         [ 0.36   ,0.37 ],
                         [ 0.593  ,0.042],
                         [ 0.719  ,0.103],
                         [ 0.359  ,0.188],
                         [ 0.339  ,0.241],
                         [ 0.282  ,0.257],
                         [ 0.748  ,0.232],
                         [ 0.714  ,0.346],
                         [ 0.483  ,0.312],
                         [ 0.478  ,0.437],
                         [ 0.525  ,0.369],
                         [ 0.751  ,0.489],
                         [ 0.532  ,0.472],
                         [ 0.473  ,0.376],
                         [ 0.725  ,0.445],
                         [ 0.446  ,0.459]])


k = 2
res,mu = kmeans(watermelon,k)
print(res)
print('新的中心：',mu)


class GaussianMixture:
    "Model mixture of two univariate Gaussians and their EM estimation"

    def __init__(self, data, mu_min=min(data), mu_max=max(data), sigma_min=.1, sigma_max=1, mix=.5):
        self.data = data
        # init with multiple gaussians
        self.one = Gaussian(uniform(mu_min, mu_max),
                            uniform(sigma_min, sigma_max))
        self.two = Gaussian(uniform(mu_min, mu_max),
                            uniform(sigma_min, sigma_max))

        # as well as how much to mix them
        self.mix = mix
        self.loglike = 0.  # = log(p = 1)

    def Estep(self):
        "Perform an E(stimation)-step, freshening up self.loglike in the process"
        # compute weights
        self.loglike = 0.  # = log(p = 1)
        for datum in self.data:
            # unnormalized weights
            wp1 = self.one.pdf(datum) * self.mix
            wp2 = self.two.pdf(datum) * (1. - self.mix)
            # compute denominator
            den = wp1 + wp2
            # normalize
            wp1 /= den
            wp2 /= den
            # add into loglike
            self.loglike += log(wp1 + wp2)
            # yield weight tuple
            yield (wp1, wp2)

    def Mstep(self, weights):
        "Perform an M(aximization)-step"
        # compute denominators
        (left, rigt) = zip(*weights)
        one_den = sum(left)
        two_den = sum(rigt)
        # compute new means
        self.one.mu = sum(w * d / one_den for (w, d) in zip(left, data))
        self.two.mu = sum(w * d / two_den for (w, d) in zip(rigt, data))
        # compute new sigmas
        self.one.sigma = sqrt(sum(w * ((d - self.one.mu) ** 2)
                                  for (w, d) in zip(left, data)) / one_den)
        self.two.sigma = sqrt(sum(w * ((d - self.two.mu) ** 2)
                                  for (w, d) in zip(rigt, data)) / two_den)
        # compute new mix
        self.mix = one_den / len(data)

    def iterate(self, N=1, verbose=False):
        "Perform N iterations, then compute log-likelihood"

    def pdf(self, x):
        return (self.mix) * self.one.pdf(x) + (1 - self.mix) * self.two.pdf(x)

    def __repr__(self):
        return 'GaussianMixture({0}, {1}, mix={2.03})'.format(self.one,
                                                              self.two,
                                                              self.mix)

    def __str__(self):
        return 'Mixture: {0}, {1}, mix={2:.03})'.format(self.one,
                                                        self.two,
                                                        self.mix)