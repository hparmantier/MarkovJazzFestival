import json
import os
import sys
helper = '/home/hparmantier/Montreux Analytics/Git/MarkovJazzFestival'
json_file = '/home/hparmantier/Montreux Analytics/Git/MarkovJazzFestival/Data/graph.json'
sys.path.append(os.path.abspath(helper))
import Neo4jInterface.read_db as reader

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
    js = json.dumps(data, json_file)
    f = open(json_file, 'w')
    f.write(js)


neo_to_json("Creep")
