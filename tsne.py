import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances


class TSNE:
    def __init__(self, n_components=2, learning_rate=0.1, n_iter=1000):
        self.n_components = n_components
        self.learning_rate = learning_rate
        self.n_iter = n_iter

    def fit_transform(self, X):
        n_samples, n_features = X.shape
        Y = np.random.randn(n_samples, self.n_components)
        for i in range(self.n_iter):
            Y -= self.learning_rate * self._calculate_gradient(X, Y)
        return Y

    def _calculate_q_matrix(self, Y, beta=1.0):
        n_samples = Y.shape[0]
        similarities = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            diff = Y - Y[i]
            distances_squared = np.sum(np.square(diff), axis=1)
            similarities[:, i] = np.exp(-distances_squared * beta)
        return similarities

    def _calculate_gradient(self, X, Y):
        gradient = np.zeros_like(Y)
        P = self._calculate_p_matrix(X)
        Q = self._calculate_q_matrix(X)
        for i in range(Y.shape[0]):
            diff = Y[i] - Y
            weighted_diff = 4 * ((P[i] - Q[i])[:, np.newaxis] * diff).T
            gradient[i] = np.sum(weighted_diff, axis=1)
        return gradient

    def _calculate_p_matrix(self, X, beta=1.0):
        n_samples = X.shape[0]
        similarities = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            diff = X - X[i]
            distances_squared = np.sum(np.square(diff), axis=1)
            similarities[:, i] = np.exp(-distances_squared * beta)
        return similarities


def generate_clusters(n_clusters, n_samples_per_cluster, n_features, cluster_std):
    clusters = []

    for _ in range(n_clusters):
        center = np.random.randn(n_features)
        cluster = center + np.random.randn(n_samples_per_cluster, n_features) * cluster_std
        clusters.append(cluster)

    return np.concatenate(clusters, axis=0)

n_clusters = 5
n_samples_per_cluster = 10
n_features = 2
cluster_std = 0.5

X = generate_clusters(n_clusters, n_samples_per_cluster, n_features, cluster_std)

print(X)
tsne = TSNE(n_components=1, learning_rate=0.1, n_iter=1000)

X_embedded = tsne.fit_transform(X)
print('X_embedded', X_embedded)

plt.scatter(X_embedded[:, 0], np.arnage_like(X_embedded))
plt.scatter(X[:, 0], X[:, 1])
plt.show()

