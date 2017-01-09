import json
import os
import sys
helper = '/home/hparmantier/Montreux Analytics/Git/MarkovJazzFestival'
json_file = '/home/hparmantier/Montreux Analytics/Git/MarkovJazzFestival/Visualization/'
sys.path.append(os.path.abspath(helper))
import Neo4jInterface.read_db as reader
import matplotlib.pyplot as plt
import networkx as nx

def neo_to_json(music):
    data = []
    G = reader.stream_built_nx(music)
    for n in G:
        d = {}
        edges = list(filter(lambda e: e[0]==n, G.edges(n, data=True)))
        neighbors = [e[1] for e in edges]
        d["name"] = str(n)
        #d["size"] = 1
        d["imports"] = neighbors#map(lambda node: str(node), neighbors)
        data.append(d)
    out = json_file+music+'.json'
    js = json.dumps(data, out)
    f = open(out, 'w')
    f.write(js)

def RG(music):
    G = reader.stream_built_nx(music)
    plt.figure(figsize=(12,12))
    pos = nx.circular_layout(G)
    nx.draw_networkx_nodes(G, pos, G.nodes(), alpha=0.8, node_size=60, with_labels=True, node_color=str(0.3))
    nx.draw_networkx_edges(G, pos, alpha=0.4)
    plt.axis('off')
    plt.title("Random Graph: 'Together'")
    plt.show()


RG("together")
