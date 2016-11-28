import numpy as np
import librosa
import networkx as nx
import graph as gr

recc_mat = gr.affinity_matrix('../data/opa.mp3')
G = gr.build_graph(recc_mat)
new = nx.DiGraph()
threshold = 0.7
for n in G:
	edges = list(filter(lambda e: e[0]==n, G.edges(n, data=True)))
	print(edges)
	edges = list(filter(lambda e: e[2]['weight'] > threshold,edges))
	
	if len(edges) != 0:
		
		weights = list(map(lambda e: e[2]['weight'], edges))
		maxi = max(weights)
		n_ = list(filter(lambda e: e[2]['weight']==maxi,edges))[0][1]
		new.add_edge(n,n_,weight=maxi)

	new.add_edge(n,n,weight=1.0)


	

nx.write_gpickle(new, '../Data/opa_nx.gpickle')