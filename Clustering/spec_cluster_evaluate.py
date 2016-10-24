from __future__ import print_function
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, metrics
import networkx as nx

def array_to_dict(arr):
    return dict(enumerate(arr))

def main():
    ########################################SPECTRAL CLUSTERING ON recurrence.npy####################################
    nb_clstr = [2,3,4] + list(range(5,35,5))
    recc_matrix = np.load('/home/hparmantier/Montreux Analytics/clustering/data/R_creep.npy')
    feat_matrix = np.load('/home/hparmantier/Montreux Analytics/clustering/data/mfcc_creep.npy').transpose()
    print(str(feat_matrix.shape))
    print(str(recc_matrix.shape))
    G = nx.from_numpy_matrix(recc_matrix)
    labels = {}
    for nc in nb_clstr:
        labels[nc] = cluster.spectral_clustering(1*recc_matrix, nc, eigen_solver='arpack')


    silhouettes = {}
    for k,v in labels.items():
        print("########## Graph with number of cluster:" + str(k) + " ###########")
        silh = metrics.silhouette_score(feat_matrix, v, metric='euclidean')
        silhouettes[k] = silh
        print(str(silh))

        d = array_to_dict(v)
        values = [d.get(node) for  node in G.nodes()]
        nx.draw_spring(G, map = plt.get_cmap('jet'), node_color= values, node_size=30, with_labels=False)
        plt.show()
        input()

    print(silhouettes.items())
