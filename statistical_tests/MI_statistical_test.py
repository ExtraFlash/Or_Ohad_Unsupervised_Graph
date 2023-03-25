import numpy as np
import pandas as pd
from utils import *
import clustering_methods
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics import adjusted_mutual_info_score
from scipy.stats import f_oneway
from scipy.stats import ttest_rel
from scipy.stats import mannwhitneyu
from scipy.stats import kruskal


def perform_anova_test():
    mi_scores_per_clustering_list = []
    for clustering_methods_name in CLUSTERING_METHODS_NAMES_LIST:
        df = pd.read_csv(f'../data/MI/clustering_methods/{clustering_methods_name}')
        mi_scores = df[TARGET].tolist()
        mi_scores_per_clustering_list.append(mi_scores)

    print(kruskal(*mi_scores_per_clustering_list))


# saves MI scores for each clustering method with the gender, in the test data
def save_scores():
    # MI_optimal_per_clustering_test_data
    df_mi_optimal_per_clustering_test = pd.DataFrame(columns=CLUSTERING_METHODS_NAMES_LIST)

    data = pd.read_csv('../data/test_data.csv')
    X_test = data.drop(TARGET, axis=1)
    y_test = data[TARGET]

    for clustering_method_name in CLUSTERING_METHODS_NAMES_LIST:
        print(f'Starting: {clustering_method_name}')
        dims_num = clustering_methods_optimal_dims_num[clustering_method_name]
        clusters_num = clustering_methods_optimal_clusters_num[clustering_method_name]

        # perform dims reduction
        ipca = IncrementalPCA(n_components=dims_num)
        ipca_test_data = ipca.fit_transform(X_test)

        # perform clustering
        clustering_function = clustering_methods_functions_dict[clustering_method_name]
        clustering_method, pred_labels_cv = clustering_function(data=ipca_test_data,
                                                                n_clusters=clusters_num)
        mi_score = adjusted_mutual_info_score(y_test, pred_labels_cv)
        df_mi_optimal_per_clustering_test.loc[0, clustering_method_name] = mi_score
    df_mi_optimal_per_clustering_test.to_csv(f'../data/MI_optimal_per_clustering_test_data')


def perform_ttest(clustering_method_name1, clustering_method_name2):
    df_scores1 = pd.read_csv(f'../data/MI/clustering_methods/{clustering_method_name1}')
    df_scores2 = pd.read_csv(f'../data/MI/clustering_methods/{clustering_method_name2}')

    scores1 = df_scores1[TARGET].tolist()
    scores2 = df_scores2[TARGET].tolist()

    print(mannwhitneyu(scores1, scores2, alternative='greater'))


if __name__ == '__main__':
    clustering_methods_optimal_dims_num = clustering_methods.CLUSTERING_METHODS_OPTIMAL_DIMS_NUM
    clustering_methods_optimal_clusters_num = clustering_methods.CLUSTERING_METHODS_OPTIMAL_CLUSTERS_NUM
    clustering_methods_functions_dict = clustering_methods.CLUSTERING_METHODS_FUNCTIONS_DICT

    perform_anova_test()
    # save_scores()
    perform_ttest('minibatchkmeans', 'gmm')
