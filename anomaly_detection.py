import pandas as pd
from utils import *
import clustering_methods
from sklearn.metrics import silhouette_score
from sklearn.decomposition import IncrementalPCA
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import adjusted_mutual_info_score


def perform_isolation_forest():
    clf = IsolationForest()
    clf.fit(X_test)
    scores = clf.predict(X_test)
    # minus = [1 for score in scores if score == -1]
    return scores


def perform_one_class_svm():
    # nu is an upper bound for the percentage of outliers in the data
    one_class_svm = OneClassSVM(gamma='scale', nu=0.01)
    one_class_svm.fit(X_test)
    scores = one_class_svm.predict(X_test)
    return scores


def perform_best_clustering_after_removing_anomalies(anomaly_pred):
    n_clusters = clustering_methods_optimal_clusters_num['minibatchkmeans']
    n_dims = clustering_methods_optimal_dims_num['minibatchkmeans']

    non_anomaly_indices = [i for i in range(len(anomaly_pred)) if anomaly_pred[i] == 1]
    non_anomaly_data = X_test.iloc[non_anomaly_indices]

    ipca = IncrementalPCA(n_components=n_dims)
    ipca_test_data = ipca.fit_transform(non_anomaly_data)

    kmeans, labels = clustering_methods.kmeans(data=ipca_test_data,
                                               n_clusters=n_clusters)
    return silhouette_score(non_anomaly_data, labels)


if __name__ == '__main__':
    data = pd.read_csv('data/test_data.csv')
    X_test = data.drop(TARGET, axis=1)
    y_test = data[TARGET]

    # arrays of: 1 if non anomaly point, -1 if anomaly point
    iso_forest_pred = perform_isolation_forest()
    one_class_pred = perform_one_class_svm()

    # indices of anomaly points
    iso_forest_anomaly_indices = [i for i in range(len(iso_forest_pred)) if iso_forest_pred[i] == -1]
    one_class_anomaly_indices = [i for i in range(len(one_class_pred)) if one_class_pred[i] == -1]

    iso_forest_anomaly_genres = y_test.iloc[iso_forest_anomaly_indices].tolist()
    one_class_anomaly_genres = y_test.iloc[one_class_anomaly_indices].tolist()
    # mutual information between anomaly labels and genres
    # iso_forest_mi = adjusted_mutual_info_score(iso_forest_anomaly_genres,
    #                                           [1 for i in range(len(iso_forest_pred)) if iso_forest_pred[i] == -1])

    # one_class_mi = adjusted_mutual_info_score(one_class_anomaly_genres,
    #                                           [1 for i in range(len(one_class_pred)) if one_class_pred[i] == -1])
    iso_forest_mi = adjusted_mutual_info_score(iso_forest_pred,
                                               y_test)

    one_class_mi = adjusted_mutual_info_score(one_class_pred,
                                              y_test)

    print(f'Iso Forest Anomaly Genres: {iso_forest_anomaly_genres}')
    print(f'Isolation Forest percentage of anomalies: {len(iso_forest_anomaly_indices)} / {X_test.shape[0]}')
    print(f'One Clas SVM percentage of anomalies: {len(one_class_anomaly_genres)} / {X_test.shape[0]}')
    print(f'Isolation Forest MI with genres: {iso_forest_mi}')
    print(f'One Clas SVM MI with genres: {one_class_mi}')

    # now perform the best clustering with removal of anomalies and check for new silhouette score

    # loading clustering information
    clustering_methods_optimal_dims_num = clustering_methods.CLUSTERING_METHODS_OPTIMAL_DIMS_NUM
    clustering_methods_optimal_clusters_num = clustering_methods.CLUSTERING_METHODS_OPTIMAL_CLUSTERS_NUM

    iso_forest_silhouette = perform_best_clustering_after_removing_anomalies(iso_forest_pred)
    one_class_silhouette = perform_best_clustering_after_removing_anomalies(one_class_pred)

    print(f'Silhouette after Isolation Forest: {iso_forest_silhouette}')
    print(f'Silhouette after OneClassSVM: {one_class_silhouette}')
