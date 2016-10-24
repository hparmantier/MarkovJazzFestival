import numpy as np
import librosa
import networkx as nx
import graph as gr

recc_mat = gr.affinity_matrix('../data/creep.mp3')
G = gr.build_graph(recc_mat)
nx.write_gpickle(G, '../data/test.gpickle')