from __future__ import print_function
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, metrics
import networkx as nx
from spec_cluster_evaluate import array_to_dict

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

def main():
    ########################################RECURRENCE MATRIX USING LIBROSA##########################################
    # audio_file = '/home/hparmantier/Montreux Analytics/come_tog.wav'
    # y, sr = librosa.load(audio_file)
    # mfcc = librosa.feature.mfcc(y=y[0:50000], sr=sr)
    # width, height = mfcc.shape
    # recc_matrix = librosa.segment.recurrence_matrix(mfcc, mode='connectivity')
    #
    # # draw recurrence matrix
    # #librosa.display.specshow(recc_matrix[0:199, 0:199], x_axis='time', y_axis='time', aspect='equal')
    # #plt.title('Binary recurrence (connectivity)')
    # #plt.show()
    ########################################SPECTRAL CLUSTERING ON recurrence.npy####################################
    n1,n2 = (2,3)
    recc_matrix = np.load('/home/hparmantier/Montreux Analytics/clustering/data/R_creep.npy')
    G = nx.from_numpy_matrix(recc_matrix)
    labels = cluster.spectral_clustering(recc_matrix, n1, eigen_solver='arpack')
    draw_temporal_labels(labels)
