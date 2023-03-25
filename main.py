# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import json

import pandas as pd
import sklearn as sk
import networkx as nx
from sklearn.utils import shuffle
from utils import *


# returns normalized sorted degrees vector of size 11 (in descending order)
def getDegreesVector(G, size=11, weight: float = 1):
    degrees = [val for (node, val) in G.degree()]
    degrees.sort(reverse=True)
    degrees_np = np.array(degrees)[:size]

    current_size = degrees_np.shape[0]
    if current_size < size:
        con = np.concatenate((degrees_np, np.zeros(size - current_size)), axis=0)
        return weight * con / np.linalg.norm(con)
    return weight * degrees_np / np.linalg.norm(degrees_np)

# returns normalized sorted eigenvalues vector of size 11 (in descending order)
def getEigenvaluesVector(G, size=11, weight: float = 1):
    eigs = nx.normalized_laplacian_spectrum(G)
    eigs = -np.sort(-eigs)
    eigs = weight * eigs[:size]

    current_size = eigs.shape[0]
    if current_size < size:
        con = np.concatenate((eigs, np.zeros(size - current_size)), axis=0)
        return weight * con / np.linalg.norm(con)
    return weight * eigs / np.linalg.norm(eigs)

# returns normalized sorted betweenness centrality vector of size 11 (in descending order)
def getBetweennessCentrality(G, size=11, weight: float = 1):
    bet = nx.betweenness_centrality(G)
    bet_list = list(bet.values())
    bet_list.sort(reverse=True)
    bet_np = np.array(bet_list)[:size]

    current_size = bet_np.shape[0]
    if current_size < size:
        con = np.concatenate((bet_np, np.zeros(size - current_size)), axis=0)
        return weight * con / np.linalg.norm(con)
    return weight * bet_np / np.linalg.norm(bet_np)


def getAverageClusteringCoefficient(G):
    return nx.average_clustering(G)


def getEigenvectorCentrality(G, size=11, weight: float = 1):
    centrality = list(nx.eigenvector_centrality(G).values())
    centrality.sort()
    centrality_np = np.array(centrality)[:size]

    current_size = centrality_np.shape[0]
    if current_size < size:
        con = np.concatenate((centrality_np, np.zeros(size - current_size)), axis=0)
        return weight * con / np.linalg.norm(con)
    return weight * centrality_np / np.linalg.norm(centrality_np)


def getClosenessCentrality(G, size=11, weight: float = 1):
    centrality = list(nx.closeness_centrality(G).values())
    centrality.sort()
    centrality_np = np.array(centrality)[:size]

    current_size = centrality_np.shape[0]
    if current_size < size:
        con = np.concatenate((centrality_np, np.zeros(size - current_size)), axis=0)
        return weight * con / np.linalg.norm(con)
    return weight * centrality_np / np.linalg.norm(centrality_np)


def getDegreeCentrality(G, size=11, weight: float = 1):
    centrality = list(nx.degree_centrality(G).values())
    centrality.sort()
    centrality_np = np.array(centrality)[:size]

    current_size = centrality_np.shape[0]
    if current_size < size:
        con = np.concatenate((centrality_np, np.zeros(size - current_size)), axis=0)
        return weight * con / np.linalg.norm(con)
    return weight * centrality_np / np.linalg.norm(centrality_np)


def getDiameter(G):
    return nx.diameter(G)


def creatingEgoVectorsDataframe():
    f = open('data/deezer_edges.json')
    # dictionary of graphs (key: index, value: graph edges)
    data = json.load(f)

    ego_networks = []  # list of 9628 networkx graphs
    for i in range(AMOUNT_OF_GRAPHS):
        ego_networks.append(nx.Graph(data[f'{i}']))

    # each row represents an ego network, containing 68 features extracted, and another one is gender label, 0 or 1
    columns = list(range(51))
    columns.append('target')
    df = pd.DataFrame(index=range(AMOUNT_OF_GRAPHS), columns=columns)
    deezer_target = pd.read_csv('data/deezer_target.csv')
    for i in range(AMOUNT_OF_GRAPHS):
        print(f'Iteration {i}')
       # degreesVector = getDegreesVector(ego_networks[i], size=20)
        eigenvaluesVector = getEigenvaluesVector(ego_networks[i], size=50)
      #  betweennessCentralityVector = getBetweennessCentrality(ego_networks[i], size=100, weight=1)
        average_clustering_coefficient = np.array([getAverageClusteringCoefficient(ego_networks[i])])
      #  eigenvector_centrality = getEigenvectorCentrality(ego_networks[i], size=50)
     #   closeness_centrality = getClosenessCentrality(ego_networks[i], size=50)
     #   degree_centrality = getDegreeCentrality(ego_networks[i], size=50)

        targetVector = np.array([deezer_target.loc[i, 'target']])
        targetVector = targetVector.astype(int)

        df.iloc[i] = np.concatenate((eigenvaluesVector, average_clustering_coefficient, targetVector),
                                    axis=0)
    # shuffling the data
    df = shuffle(df, random_state=42)
    df.reset_index(inplace=True, drop=True)
    df.to_csv('data/ego_vectors_data')
    f.close()


if __name__ == '__main__':
    creatingEgoVectorsDataframe()
    ego_vectors_data = pd.read_csv('data/ego_vectors_data', index_col=0)

    # splitting to train data and test data
    train_data = ego_vectors_data.iloc[:TRAIN_DATA_SIZE]
    test_data = ego_vectors_data.iloc[TRAIN_DATA_SIZE:]

    print(f'train data: {train_data.shape}')
    print(f'test data: {test_data.shape}')

    train_data.to_csv('data/train_data.csv')
    test_data.to_csv('data/test_data.csv')