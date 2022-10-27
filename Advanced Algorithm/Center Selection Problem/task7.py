import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.datasets import make_blobs
#  pip install scikit-fuzzy scikit-learn-extra
from sklearn_extra.cluster import KMedoids
from skfuzzy.cluster import cmeans


def task7_1():
    # 直接使用k-means++ 初始化算法
    print('task7-1')
    fig_row = 1
    fig_col = 1
    fig = plt.figure(dpi=300)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
    # original data
    ax = fig.add_subplot(fig_row, fig_col, 1)
    row, _ = np.shape(X)
    for i in range(row):
        ax.plot(X[i, 0], X[i, 1], '#4EACC5', marker='.')

    ax.set_title('Data Set')
    ax.set_xticks(())
    ax.set_yticks(())
    plt.show()


def task7_2():
    # compute clustering with K-Means
    fig = plt.figure(figsize=(8, 4), dpi=300)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)

    fig_row = 1
    fig_col = 2

    # compute clustering with K-Means
    k_means = KMeans(init='k-means++', n_clusters=n_clusters)
    k_means.fit(X)

    k_means_cluster_centers = np.sort(k_means.cluster_centers_, axis=0)
    k_means_labels = pairwise_distances_argmin(X, k_means_cluster_centers)

    # K-means
    ax = fig.add_subplot(fig_row, fig_col, 1)
    for k, col in zip(range(n_clusters), colors):
        my_members = k_means_labels == k  # my_members是布尔型的数组（用于筛选同类的点，用不同颜色表示）
        cluster_center = k_means_cluster_centers[k]
        ax.plot(X[my_members, 0], X[my_members, 1], 'w',
                markerfacecolor=col, marker='.')  # 将同一类的点表示出来
        ax.plot(cluster_center[0], cluster_center[1], markerfacecolor=col,
                markeredgecolor='k', marker='o')  # 将聚类中心单独表示出来
    ax.set_title('k-means')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_aspect(True)

    # compute clustering with k_medoids
    k_medoids = KMedoids(n_clusters=n_clusters)
    k_medoids.fit(X)

    k_medoids_cluster_centers = np.sort(k_medoids.cluster_centers_, axis=0)
    k_medoids_labels = pairwise_distances_argmin(X, k_medoids_cluster_centers)

    # k_medoids
    ax = fig.add_subplot(fig_row, fig_col, 2)
    for k, col in zip(range(n_clusters), colors):
        my_members = k_medoids_labels == k  # my_members是布尔型的数组（用于筛选同类的点，用不同颜色表示）
        cluster_center = k_medoids_cluster_centers[k]
        ax.plot(X[my_members, 0], X[my_members, 1], 'w',
                markerfacecolor=col, marker='.')  # 将同一类的点表示出来
        ax.plot(cluster_center[0], cluster_center[1], markerfacecolor=col,
                markeredgecolor='k', marker='o')  # 将聚类中心单独表示出来
    ax.set_title('k-medoids')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_aspect(True)
    plt.show()


def task7_3():
    fig = plt.figure(dpi=300)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
    fig_row = 2
    fig_col = 3

    # compute clustering with K-Means
    k_means = KMeans(init='k-means++', n_clusters=n_clusters)
    k_means.fit(X)

    k_means_cluster_centers = np.sort(k_means.cluster_centers_, axis=0)
    k_means_labels = pairwise_distances_argmin(X, k_means_cluster_centers)

    # K-means
    ax = fig.add_subplot(fig_row, fig_col, 1)
    for k, col in zip(range(n_clusters), colors):
        my_members = k_means_labels == k  # my_members是布尔型的数组（用于筛选同类的点，用不同颜色表示）
        cluster_center = k_means_cluster_centers[k]
        ax.plot(X[my_members, 0], X[my_members, 1], 'w',
                markerfacecolor=col, marker='.')  # 将同一类的点表示出来
        ax.plot(cluster_center[0], cluster_center[1], markerfacecolor=col,
                markeredgecolor='k', marker='o')  # 将聚类中心单独表示出来
    ax.set_title('k-means')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_aspect(True)

    mLists = [1.01, 5, 10, 20, 40]
    cnt = 2
    for m in mLists:
        # compute clustering with cmeans
        center, u, u0, d, jm, p, fpc = cmeans(X.T, m=m, c=n_clusters, error=0.005, maxiter=1000)
        c_means_cluster_centers = center
        c_means_label = np.argmax(u, axis=0)

        # K-means
        ax = fig.add_subplot(fig_row, fig_col, cnt)
        for k, col in zip(range(n_clusters), colors):
            my_members = c_means_label == k  # my_members是布尔型的数组（用于筛选同类的点，用不同颜色表示）
            cluster_center = c_means_cluster_centers[k]
            ax.plot(X[my_members, 0], X[my_members, 1], 'w',
                    markerfacecolor=col, marker='.')  # 将同一类的点表示出来
            ax.plot(cluster_center[0], cluster_center[1], markerfacecolor=col,
                    markeredgecolor='k', marker='o')  # 将聚类中心单独表示出来
        ax.set_title(f'm ={m}')
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_aspect(True)
        cnt += 1

    plt.suptitle(f'task7-3: n={n_samples}, k={n_clusters}')
    plt.show()


def task7_4():
    fig = plt.figure(figsize=(8, 4), dpi=300)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
    fig_row = 1
    fig_col = 2

    # compute clustering with K-Means
    k_means = KMeans(init='k-means++', n_clusters=n_clusters)
    k_means.fit(X)

    k_means_cluster_centers = np.sort(k_means.cluster_centers_, axis=0)
    k_means_labels = pairwise_distances_argmin(X, k_means_cluster_centers)

    # K-means
    ax = fig.add_subplot(fig_row, fig_col, 1)
    for k, col in zip(range(n_clusters), colors):
        my_members = k_means_labels == k  # my_members是布尔型的数组（用于筛选同类的点，用不同颜色表示）
        cluster_center = k_means_cluster_centers[k]
        ax.plot(X[my_members, 0], X[my_members, 1], 'w',
                markerfacecolor=col, marker='.')  # 将同一类的点表示出来
        ax.plot(cluster_center[0], cluster_center[1], markerfacecolor=col,
                markeredgecolor='k', marker='o')  # 将聚类中心单独表示出来
    ax.set_title('k-means')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_aspect(True)

    # compute clustering with cmeans
    center, u, u0, d, jm, p, fpc = cmeans(X.T, m=c_means_m, c=n_clusters, error=0.005, maxiter=1000)
    c_means_cluster_centers = center
    c_means_label = np.argmax(u, axis=0)

    # c-means
    ax = fig.add_subplot(fig_row, fig_col, 2)
    for k, col in zip(range(n_clusters), colors):
        my_members = c_means_label == k  # my_members是布尔型的数组（用于筛选同类的点，用不同颜色表示）
        cluster_center = c_means_cluster_centers[k]
        ax.plot(X[my_members, 0], X[my_members, 1], 'w',
                markerfacecolor=col, marker='.')  # 将同一类的点表示出来
        ax.plot(cluster_center[0], cluster_center[1], markerfacecolor=col,
                markeredgecolor='k', marker='o')  # 将聚类中心单独表示出来
    ax.set_title(f'fuzzy c-means m={c_means_m}')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_aspect(True)

    plt.suptitle(f'task7-4: n={n_samples}, k={n_clusters}')
    plt.show()


if __name__ == '__main__':
    fig_row = 1
    fig_col = 2
    n_samples = 1000
    n_clusters = 4
    np.random.seed(int(time.time()))
    c_means_m = 1.01
    # 分别对应, 波尔多红,普鲁士蓝,木乃伊棕,蒂芙尼蓝,榄菜紫, 薄荷绿
    colors = ['#4EACC5', '#FF9C34', '#4E9A06', '#4C0009', '#003153', '#8f4b28', '#0abab5', '#8e2961', '#16982b']

    centers = [[1, 1], [-1, -1], [1, -1]]
    X, labels_true = make_blobs(n_samples=n_samples, centers=centers, cluster_std=0.7)

    # plot result
    task7_2()
    # task7_3()
    # task7_4()
