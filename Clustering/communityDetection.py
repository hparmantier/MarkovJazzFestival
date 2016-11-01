import numpy as np
import community
import networkx as nx
import matplotlib.pyplot as plt


def partition(G):
    return community.best_partition(G)


def draw_partition(part, G):
    values = [part.get(node) for  node in G.nodes()]
    nx.draw_spring(G, map = plt.get_cmap('jet'), node_color= values, node_size=30, with_labels=False)
    plt.show()


def test():
    recc_matrix = np.load('/home/hparmantier/Montreux Analytics/clustering/data/R_creep.npy')
    G = nx.from_numpy_matrix(recc_matrix)
    part = community.best_partition(G)
    print(part)
