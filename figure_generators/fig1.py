import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from utils import *
import clustering_methods
from sklearn.metrics import silhouette_score
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics import silhouette_samples
from matplotlib.ticker import FixedLocator, FixedFormatter

from matplotlib import style

matplotlib.rc('font', family='Times New Roman')


def plot_bar(data, x_label, y_label, title, bottom, upperBound: float = 1):
    #style.use('ggplot')

    #plt.figure(figsize=(8, 11))
    barWidth = 0.6

    left = np.array(range(4))
    height = []
    for clustering_method_name in CLUSTERING_METHODS_NAMES_LIST:
        score = data.loc[0, clustering_method_name]
        height.append(score - bottom)

    #ax.ylim([bottom, 1])
    plt.ylim([bottom, upperBound])

    tick_label = []
    for clustering_method_name in CLUSTERING_METHODS_NAMES_LIST:
        tick_label.append(CLUSTERING_METHODS_PLOT_NAMES_DICT[clustering_method_name])

    plt.bar(left, height, color=CLUSTERING_COLORS, width=barWidth, bottom=bottom, tick_label=tick_label)

    if x_label:
        plt.xlabel(x_label, fontsize=13)
    # naming the y-axis
    plt.ylabel(y_label, fontsize=13)
    # plot title
    plt.title(title, fontsize=13)
    #ax.savefig(f'../figures/mean_silhouette_MI_optimal_per_cluster', pad_inches=0.2, bbox_inches="tight")
    #ax.show()


def silhouette_per_point_plot():

    n_clusters = clustering_methods_optimal_clusters_num['kmeans']
    n_dims = clustering_methods_optimal_dims_num['kmeans']

    test_data = pd.read_csv('../data/test_data.csv')
    X_test = test_data.drop(TARGET, axis=1)

    ipca = IncrementalPCA(n_components=n_dims)
    ipca_test_data = ipca.fit_transform(X_test)

    kmeans, y_pred = clustering_methods.kmeans(data=ipca_test_data,
                                                   n_clusters=n_clusters)
    silhouette_coefficients = silhouette_samples(ipca_test_data, y_pred)
    silhouette_score_ = silhouette_score(ipca_test_data, y_pred)

    padding = len(ipca_test_data) // 30
    pos = padding
    ticks = []
    for i in range(n_clusters):
        coeffs = silhouette_coefficients[y_pred == i]
        coeffs.sort()

        color = matplotlib.cm.Spectral(i / n_clusters)

        plt.fill_betweenx(np.arange(pos, pos + len(coeffs)), 0, coeffs,
                      facecolor=color, edgecolor=color, alpha=0.7)
        ticks.append(pos + len(coeffs) // 2)
        pos += len(coeffs) + padding

    plt.gca().yaxis.set_major_locator(FixedLocator(ticks))
    plt.gca().yaxis.set_major_formatter(FixedFormatter(range(n_clusters)))
    plt.ylabel("Cluster", fontsize=13)

    plt.gca().set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.xlabel("Silhouette Coefficient", fontsize=13)

    plt.axvline(x=silhouette_score_, color="red", linestyle="--")
    #plt.title("$Number of clusters={}$".format(n_clusters), fontsize=16)
    plt.title("D", fontsize=13)






if __name__ == '__main__':
    clustering_methods_optimal_dims_num = clustering_methods.CLUSTERING_METHODS_OPTIMAL_DIMS_NUM
    clustering_methods_optimal_clusters_num = clustering_methods.CLUSTERING_METHODS_OPTIMAL_CLUSTERS_NUM


    fig = plt.figure(figsize=(13, 10))
    #a1 = plt.subplot2grid((2, 2), (0, 0))
    #a2 = plt.subplot2grid((2, 2), (0, 1))
    #a3 = plt.subplot2grid((2, 2), (1, 0))
    #a4 = plt.subplot2grid((2, 2), (1, 1))

    # plot silhouette score for every clustering method
    plt.subplot(221)
    plot_data = pd.read_csv('../data/silhouette_optimal_per_clustering_test_data', index_col=0)
    plot_bar(plot_data, x_label="", y_label='Silhouette score',
             title='A', bottom=-1)

    # plot MI with clustering methods and genres
    plt.subplot(222)
    plot_data = pd.read_csv('../data/MI_optimal_per_clustering_test_data', index_col=0)
    plot_bar(plot_data, x_label="", y_label='MI score',
             title='B', bottom=0, upperBound=0.002)

    # plot weighted score with clustering methods and genres
    plt.subplot(223)
    plot_data = pd.read_csv('../data/mean_silhouette_MI_optimal_per_clustering_test_data', index_col=0)
    plot_bar(plot_data, x_label="", y_label='Weighted Silhouette and MI',
             title='C', bottom=0)
    fig.tight_layout(pad=2)

    plt.subplot(224)
    silhouette_per_point_plot()
    plt.savefig('../figures/Fig1', pad_inches=0.2, bbox_inches="tight")
    plt.show()