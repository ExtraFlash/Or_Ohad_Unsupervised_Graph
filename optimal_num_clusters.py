import pandas as pd
import clustering_methods
from utils import *
from sklearn.metrics import silhouette_score


def find_num_of_clusters_silhouette(clustering_method_name):
    max_score = 0
    best_clusters_num = 0
    df = pd.DataFrame(columns=clusters_num_list)
    for clusters_num in clusters_num_list:
        print(f'{clustering_method_name}, clusters amount: {clusters_num}')

        clustering_function = clustering_methods_functions_dict[clustering_method_name]
        clustering_method, labels = clustering_function(X, n_clusters=clusters_num)
        score = silhouette_score(X, labels)
        if score > max_score:
            max_score = score
            best_clusters_num = clusters_num
        df.loc[0, clusters_num] = score
    df.to_csv(f'data/silhouette_best_clusters_num_train_data/{clustering_method_name}')
    print(f'Best amount of clusters for {clustering_method_name} is: {best_clusters_num}, score: {max_score}')


if __name__ == '__main__':
    data = pd.read_csv('data/train_data.csv')
    X = data.drop(TARGET, axis=1)
    print(X.head())

    clusters_num_list = list(range(2, CLUSTERS_MAX_AMOUNT + 1))  # from 2 to the max number of clusters

    clustering_methods_functions_dict = clustering_methods.CLUSTERING_METHODS_FUNCTIONS_DICT

    # find optimal number of clusters for all clustering methods
    for clustering_method_name_ in CLUSTERING_METHODS_NAMES_LIST:
        find_num_of_clusters_silhouette(clustering_method_name_)