import pandas as pd
from utils import *
import clustering_methods
from sklearn.metrics import silhouette_score
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics import adjusted_mutual_info_score
from scipy.stats import f_oneway
from scipy.stats import ttest_rel


def save_mean_silhouette_MI_cvs_scores():
    df_silhouette_cvs_scores = pd.read_csv('../data/silhouette_cvs_for_anova_test_data')
    df_mean_silhouette_MI_scores = pd.DataFrame(index=list(range(cv_amount)), columns=CLUSTERING_METHODS_NAMES_LIST)

    for clustering_methods_name in CLUSTERING_METHODS_NAMES_LIST:
        df_MI_silhouette_cvs_scores = pd.read_csv(f'../data/MI/clustering_methods/{clustering_methods_name}')
        for i in range(cv_amount):
            sil_score = df_silhouette_cvs_scores.loc[i, clustering_methods_name]
            MI_score = df_MI_silhouette_cvs_scores.loc[0, TARGET]
            mean_score = 0.7 * sil_score + 0.3 * MI_score
            df_mean_silhouette_MI_scores.loc[i, clustering_methods_name] = mean_score

    df_mean_silhouette_MI_scores.to_csv(f'../data/mean_silhouette_MI_cvs_scores_test_data')


def save_mean_silhouette_MI_scores():
    df_mean_silhouette_MI_scores = pd.DataFrame(columns=CLUSTERING_METHODS_NAMES_LIST)
    for clustering_method_name in CLUSTERING_METHODS_NAMES_LIST:
        dims_num = clustering_methods_optimal_dims_num[clustering_method_name]

        # perform dims reduction
        ipca = IncrementalPCA(n_components=dims_num)
        ipca_test_data = ipca.fit_transform(X_test)

        # perform clustering
        clustering_function = clustering_methods_functions_dict[clustering_method_name]
        clustering_method, pred_labels = clustering_function(data=ipca_test_data,
                                                             n_clusters=clusters_num)
        mi_score = adjusted_mutual_info_score(pred_labels, y_test)
        sil_score = silhouette_score(X_test, pred_labels)
        mean_score = sil_score * 0.7 + mi_score * 0.3
        df_mean_silhouette_MI_scores.loc[0, clustering_method_name] = mean_score

    df_mean_silhouette_MI_scores.to_csv('../data/mean_silhouette_MI_optimal_per_clustering_test_data')


def perform_anova_test():
    scores_df = pd.read_csv('../data/mean_silhouette_MI_cvs_scores_test_data')

    # list of lists of cvs per clustering method
    scores_cvs_data = []

    for clustering_method_name in CLUSTERING_METHODS_NAMES_LIST:
        clustering_scores = scores_df[clustering_method_name].tolist()
        scores_cvs_data.append(clustering_scores)

    print(f_oneway(*scores_cvs_data))


def perform_ttest(clustering_method_name1, clustering_method_name2):
    scores_df = pd.read_csv('../data/mean_silhouette_MI_cvs_scores_test_data')
    scores1 = scores_df[clustering_method_name1]
    scores2 = scores_df[clustering_method_name2]

    print(ttest_rel(scores1, scores2, alternative='greater'))


if __name__ == '__main__':
    cv_amount = 40
    clusters_num = CLASSES_AMOUNT

    data = pd.read_csv('../data/test_data.csv')
    X_test = data.drop(TARGET, axis=1)
    y_test = data[TARGET]

    clustering_methods_optimal_dims_num = clustering_methods.CLUSTERING_METHODS_OPTIMAL_DIMS_NUM
    clustering_methods_optimal_clusters_num = clustering_methods.CLUSTERING_METHODS_OPTIMAL_CLUSTERS_NUM
    clustering_methods_functions_dict = clustering_methods.CLUSTERING_METHODS_FUNCTIONS_DICT

    #save_mean_silhouette_MI_cvs_scores()

    #save_mean_silhouette_MI_scores()

    #perform_anova_test()

    perform_ttest('kmeans', 'minibatchkmeans')