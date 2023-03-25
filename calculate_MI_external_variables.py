import pandas as pd
from utils import *
import clustering_methods
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics import adjusted_mutual_info_score


def calc_MI(clustering_method_name):
    df = pd.DataFrame(index=list(range(cv_amount)), columns=EXTERNAL_VARIABLES_NAMES)
    cvs = get_cvs(n_rows=X_test.shape[0], cv=cv_amount)
    for i, cv in enumerate(cvs):
        X_cv = X_test.iloc[cv]
        y_cv = y_test.iloc[cv]
        print(f'{clustering_method_name}, cv: {i}')

        dims_num = clustering_methods_optimal_dims_num[clustering_method_name]

        # perform dims reduction
        ipca = IncrementalPCA(n_components=dims_num)
        ipca_test_data = ipca.fit_transform(X_cv)

        # perform clustering
        clustering_function = clustering_methods_functions_dict[clustering_method_name]
        clustering_method, pred_labels_cv = clustering_function(data=ipca_test_data,
                                                                n_clusters=clusters_num)
        for external_variable_name in EXTERNAL_VARIABLES_NAMES:
            true_labels_cv = y_cv[external_variable_name]
            # print(type(true_labels_cv))
            # print(type(pred_labels))
            # print(f'true labels: {true_labels_cv}')
            # print(f'pred labels: {pred_labels}')
            mi_score = adjusted_mutual_info_score(true_labels_cv, pred_labels_cv)
            df.loc[i, external_variable_name] = mi_score

    df.to_csv(f'{save_dir}/{clustering_method_name}')


if __name__ == '__main__':
    cv_amount = 40
    data = pd.read_csv('data/test_data.csv')
    X_test = data.drop(EXTERNAL_VARIABLES_NAMES, axis=1)
    y_test = data[EXTERNAL_VARIABLES_NAMES]
    save_dir = 'data/MI/clustering_methods'

    clusters_num = CLASSES_AMOUNT

    clustering_methods_optimal_dims_num = clustering_methods.CLUSTERING_METHODS_OPTIMAL_DIMS_NUM
    clustering_methods_functions_dict = clustering_methods.CLUSTERING_METHODS_FUNCTIONS_DICT

    for clustering_method_name_ in CLUSTERING_METHODS_NAMES_LIST:
        calc_MI(clustering_method_name_)