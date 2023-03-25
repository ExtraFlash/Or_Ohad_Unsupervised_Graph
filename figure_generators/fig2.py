import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from utils import *
import clustering_methods
from sklearn.decomposition import IncrementalPCA
from sklearn.manifold import TSNE
import random

from matplotlib import style

matplotlib.rc('font', family='Times New Roman')
import matplotlib.cm as cm


def plot_2d():
    n_clusters = CLASSES_AMOUNT
    n_dims = clustering_methods_optimal_dims_num['kmeans']

    ipca = IncrementalPCA(n_components=n_dims)
    ipca_test_data = ipca.fit_transform(X_test)

    minibatch, labels = clustering_methods.kmeans(data=ipca_test_data,
                                                           n_clusters=n_clusters)
    print(set(labels))
    # [green, yellow, ...]

    #colors_different_genres = cm.rainbow(np.linspace(0, 1, utils.CATEGORIES_AMOUNT))
    colors_different_genders = ['green', 'darkorange']

    clustering_different_colors = ['lime', 'moccasin']
    colors_genders_dict = {}
    for i, gender in enumerate(GENDERS):
        colors_genders_dict[gender] = colors_different_genders[i]

    samples_colors_genders_list = []
    for gender in y_test.tolist():
        color = colors_genders_dict[gender]
        samples_colors_genders_list.append(color)



    data_2d = TSNE(n_components=2).fit_transform(ipca_test_data)

    #clustering_different_colors = ["red", "blue", "green", "magenta"]
    clustering_colors_dict = {key:val for key,val in enumerate(clustering_different_colors)}

    samples_colors_clustering_list = []
    for label in labels:
        color = clustering_colors_dict[label]
        samples_colors_clustering_list.append(color)

    plt.scatter(x=data_2d[:, 0], y=data_2d[:, 1], c=samples_colors_clustering_list, marker='o', s=100)
    plt.scatter(x=data_2d[:, 0], y=data_2d[:, 1], c=samples_colors_genders_list, marker='.', s=20)
    plt.xlabel("t-SNE Component 1", fontsize=16)
    plt.ylabel("t-SNE Component 2", fontsize=16)
    plt.savefig('../figures/Fig2', pad_inches=0.2, bbox_inches="tight")
    plt.show()





if __name__ == '__main__':
    data = pd.read_csv('../data/test_data.csv')
    #sample_idx = random.sample(range(data.shape[0]), 5000) # sample 200 unique indices for sampling
    X_test = data.drop(TARGET, axis=1)
    y_test = data[TARGET]
    #X_test = X_test.iloc[sample_idx]
    #y_test = y_test.iloc[sample_idx]
    #X_test = X_test.iloc[list(range(200))]
    #y_test = y_test.iloc[list(range(200))]

    clustering_methods_optimal_dims_num = clustering_methods.CLUSTERING_METHODS_OPTIMAL_DIMS_NUM
    clustering_methods_optimal_clusters_num = clustering_methods.CLUSTERING_METHODS_OPTIMAL_CLUSTERS_NUM

    fig = plt.figure(figsize=(10, 10))

    plot_2d()