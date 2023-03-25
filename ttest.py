import numpy as np
import sklearn as skl
import pandas as pd
from sklearn.manifold import MDS
from sklearn.metrics import mean_squared_error
import os
from utils import *

#X = np.array([[0,1,2],[1,0,3],[2,3,0]])
#mds = MDS(dissimilarity='precomputed', n_components=2, random_state=42)
#X_mds = mds.fit_transform(X)
#print(X_mds)
#print(mean_squared_error(X_mds[1], X_mds[2]))

#for clustering_method_name in CLUSTERING_METHODS_NAMES_LIST:
#    path = f'data/silhouette_best_clusters_num_train_data/{clustering_method_name}'
#    if not os.path.exists(path):
#        os.makedirs(path)

data = pd.read_csv('data/test_data.csv')
X_test = data.drop(TARGET, axis=1)
y_test = data[TARGET]
y_test = y_test.iloc[[1,2,3]]
print(y_test)

