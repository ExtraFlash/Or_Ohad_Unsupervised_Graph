from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.cluster import Birch
from sklearn.cluster import MiniBatchKMeans

##### CONSTANTS

AMOUNT_OF_GRAPHS = 9629
AMOUNT_OF_FEATURES = 51
CLASSES_AMOUNT = 2
TRAIN_DATA_SIZE = 6259
TEST_DATA_SIZE = 3370

CLUSTERING_METHODS_NAMES_LIST = ['gmm', 'kmeans', 'birch', 'minibatchkmeans']

CLUSTERING_METHODS_PLOT_NAMES_DICT = {'gmm': 'GMM',
                                      'kmeans': 'KMeans',
                                      'birch': 'Birch',
                                      'minibatchkmeans': 'Minibatch KMeans '}

CLUSTERING_METHODS_DICT = {'gmm': GaussianMixture(),
                           'kmeans': KMeans(),
                           'birch': Birch(),
                           'minibatchkmeans': MiniBatchKMeans()}

CLUSTERING_COLORS = ['royalblue', 'slategrey', 'limegreen', 'deeppink']

ANOMALY_DETECTION_MODELS = ['isolation_forest']

GENDERS = [0, 1]

##### LABEL NAMES
EXTERNAL_VARIABLES_NAMES = ['target']
TARGET = 'target'

CLUSTERS_MAX_AMOUNT = 10


### FUNCTIONS

def get_cvs(n_rows, cv=10):
    cvs = []
    increment = n_rows // cv
    current_end = increment
    for i in range(cv):
        index_list = list(range(current_end - increment, current_end))
        cvs.append(index_list)
        current_end += increment
    return cvs