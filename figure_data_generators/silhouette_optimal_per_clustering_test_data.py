import pandas as pd
from utils import *
import clustering_methods
from sklearn.metrics import silhouette_score
from sklearn.decomposition import IncrementalPCA


def silhouette_for_clustering():
    df = pd.DataFrame(columns=CLUSTERING_METHODS_NAMES_LIST)

    for clustering_method_name in CLUSTERING_METHODS_NAMES_LIST:
        print(f'Starting: {clustering_method_name}')
        dims_num = clustering_methods_optimal_dims_num[clustering_method_name]
        clusters_num = clustering_methods_optimal_clusters_num[clustering_method_name]

        # perform dims reduction
        ipca = IncrementalPCA(n_components=dims_num)
        ipca_test_data = ipca.fit_transform(X_test)

        # perform clustering
        clustering_function = clustering_methods_functions_dict[clustering_method_name]
        clustering_method, labels = clustering_function(data=ipca_test_data,
                                                        n_clusters=clusters_num)
        score = silhouette_score(X_test, labels)
        df.loc[0, clustering_method_name] = score
    df.to_csv(f'../data/silhouette_optimal_per_clustering_test_data')


if __name__ == '__main__':
    data = pd.read_csv('../data/test_data.csv')
    X_test = data.drop(TARGET, axis=1)
    # y = data[[utils.GENRE_TOP_NAME]]

    clustering_methods_optimal_dims_num = clustering_methods.CLUSTERING_METHODS_OPTIMAL_DIMS_NUM
    clustering_methods_optimal_clusters_num = clustering_methods.CLUSTERING_METHODS_OPTIMAL_CLUSTERS_NUM
    clustering_methods_functions_dict = clustering_methods.CLUSTERING_METHODS_FUNCTIONS_DICT

    silhouette_for_clustering()