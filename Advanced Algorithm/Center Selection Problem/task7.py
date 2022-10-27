import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.datasets import make_blobs
from sklearn_extra.cluster import KMedoids
from skfuzzy.cluster import cmeans

# ######################################
# Generate sample data

fig_row = 2
fig_col = 2
n_samples = 800
n_clusters = 3
np.random.seed(0)

batch_size = 45
centers = [[1, 1], [-1, -1], [1, -1]]
n_clusters = len(centers)
X, labels_true = make_blobs(n_samples=n_samples, centers=centers, cluster_std=0.7)

# plot result
fig = plt.figure(figsize=(8, 8))
fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
colors = ['#4EACC5', '#FF9C34', '#4E9A06']

# original data
ax = fig.add_subplot(fig_row, fig_row, 1)
row, _ = np.shape(X)
for i in range(row):
    ax.plot(X[i, 0], X[i, 1], '#4EACC5', marker='.')

ax.set_title('Original Data')
ax.set_xticks(())
ax.set_yticks(())

# compute clustering with K-Means
k_means = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
t0 = time.time()
k_means.fit(X)
t_batch = time.time() - t0

k_means_cluster_centers = np.sort(k_means.cluster_centers_, axis=0)
k_means_labels = pairwise_distances_argmin(X, k_means_cluster_centers)

# K-means
ax = fig.add_subplot(fig_row, fig_col, 2)
for k, col in zip(range(n_clusters), colors):
    my_members = k_means_labels == k  # my_members是布尔型的数组（用于筛选同类的点，用不同颜色表示）
    cluster_center = k_means_cluster_centers[k]
    ax.plot(X[my_members, 0], X[my_members, 1], 'w',
            markerfacecolor=col, marker='.')  # 将同一类的点表示出来
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
            markeredgecolor='k', marker='o')  # 将聚类中心单独表示出来
ax.set_title('KMeans')
ax.set_xticks(())
ax.set_yticks(())
# plt.text(-3.5, 1.8, 'train time: %.2fs\ninertia: %f' % (t_batch, k_means.inertia_))

# compute clustering with k_medoids
k_medoids = KMedoids(n_clusters=n_clusters)
t0 = time.time()
k_medoids.fit(X)
t_batch = time.time() - t0

k_medoids_cluster_centers = np.sort(k_medoids.cluster_centers_, axis=0)
k_medoids_labels = pairwise_distances_argmin(X, k_medoids_cluster_centers)

# k_medoids
ax = fig.add_subplot(fig_row, fig_row, 3)
for k, col in zip(range(n_clusters), colors):
    my_members = k_medoids_labels == k  # my_members是布尔型的数组（用于筛选同类的点，用不同颜色表示）
    cluster_center = k_medoids_cluster_centers[k]
    ax.plot(X[my_members, 0], X[my_members, 1], 'w',
            markerfacecolor=col, marker='.')  # 将同一类的点表示出来
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
            markeredgecolor='k', marker='o')  # 将聚类中心单独表示出来
ax.set_title('KMedoids')
ax.set_xticks(())
ax.set_yticks(())
# plt.text(-3.5, 1.8, 'train time: %.2fs\ninertia: %f' % (t_batch, k_medoids.inertia_))

# compute clustering with cmeans
center, u, u0, d, jm, p, fpc = cmeans(X.T, m=2, c=n_clusters, error=0.005, maxiter=1000)
t0 = time.time()
t_batch = time.time() - t0

c_means_cluster_centers = center
c_means_label = np.argmax(u, axis=0)

# K-means
ax = fig.add_subplot(fig_row, fig_row, 4)
for k, col in zip(range(n_clusters), colors):
    my_members = c_means_label == k  # my_members是布尔型的数组（用于筛选同类的点，用不同颜色表示）
    cluster_center = c_means_cluster_centers[k]
    ax.plot(X[my_members, 0], X[my_members, 1], 'w',
            markerfacecolor=col, marker='.')  # 将同一类的点表示出来
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
            markeredgecolor='k', marker='o')  # 将聚类中心单独表示出来
ax.set_title('fuzzy c-means')
ax.set_xticks(())
ax.set_yticks(())

plt.show()
