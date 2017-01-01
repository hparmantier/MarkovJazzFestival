from __future__ import print_function
import os
import sys
sys.path.append(os.path.abspath('..'))
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, metrics
import networkx as nx
from GraphClustering.spec_cluster_evaluate import array_to_dict

def draw_graph_labels(labels):
    d = array_to_dict(labels)
    values = [d.get(node) for  node in G.nodes()]
    nx.draw_spring(G, map = plt.get_cmap('jet'), node_color= values, node_size=30, with_labels=False)
    plt.show()

def draw_temporal_labels(labels):
    t = range(len(labels))
    plt.scatter(t, labels)
    plt.ylabel('Label')
    plt.xlabel('Song Beats')
    plt.show()

def spectral_clustering(R):
    n1,n2 = (2,3)
    labels = cluster.spectral_clustering(R, 3, eigen_solver='arpack') 
    draw_temporal_labels(labels)