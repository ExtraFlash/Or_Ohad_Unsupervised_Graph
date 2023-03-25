from networkx import Graph
from scipy.spatial.distance import squareform, pdist
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import sklearn as skl
from sklearn.manifold import MDS
from sklearn.metrics import mean_squared_error
import networkx as nx
from utils import *
import json
import matplotlib.pyplot as plt

# Load the ego networks into a list of networkx graphs
import numpy as np
import networkx as nx
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances

f = open('data/deezer_edges.json')
# dictionary of graphs (key: index, value: graph edges)
data = json.load(f)
ego_networks = []  # list of 9628 networkx graphs
for i in range(AMOUNT_OF_GRAPHS):
    ego_networks.append(nx.Graph(data[f'{i}']))


# Step 2: Compute the Laplacian matrix for each graph
laplacian_matrices = []
for g in ego_networks:
    L = nx.laplacian_matrix(g)
    laplacian_matrices.append(L)

# Step 3: Compute the eigenvectors and eigenvalues of the Laplacian matrix
eigenvectors = []
for L in laplacian_matrices:
    w, v = np.linalg.eigh(L.toarray())
    eigenvectors.append(v)

# Step 4: Select the top k eigenvectors for each graph
k = 10
top_k_eigenvectors = []
for ev in eigenvectors:
    top_k = ev[:, 1:k+1]  # Select top k eigenvectors, skipping first (trivial) eigenvector
    top_k_eigenvectors.append(top_k)

# Step 5: Compute pairwise similarity between top k eigenvectors using cosine similarity
similarities = pairwise_distances(np.vstack(top_k_eigenvectors), metric='cosine')

# Step 6: Optionally, use MDS to visualize the graphs in a lower-dimensional space
from sklearn.manifold import MDS
embedding = MDS(n_components=2, dissimilarity='precomputed').fit_transform(similarities)
print(embedding)

f.close()
