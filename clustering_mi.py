import pandas as pd
import numpy as np
import sklearn as sk
import networkx as nx
from wl_kernel import *
from utils import *

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

data = pd.read_csv('data/ego_vectors_data', index_col=0)
X = data.drop('target', axis=1)
y = data.iloc[:,51].tolist()

#X = X.iloc[:200]
#y = y.iloc[:200]

#data_2d = TSNE(n_components=3).fit_transform(X)
pca = PCA(n_components=20)
pca_data = pca.fit_transform(X)
kmeans = KMeans(n_clusters=2)
kmeans.fit(pca_data)
#print(y)
print(adjusted_mutual_info_score(kmeans.labels_, y))

