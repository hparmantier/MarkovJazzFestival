import os
import sys
import json
helper = '/home/hparmantier/Montreux Analytics/Git/MarkovJazzFestival/RandomGraph'
sys.path.append(os.path.abspath(helper))
import intra as builder
import write_db as writer
import read_db as reader
import networkx as nx
import SimulateRW as simulator

################################################################################

def nodes2edges(nodes):
    edges = []
    i = 1
    while i != len(nodes)-1:
        edges.append((nodes[i-1], nodes[i]))
        i += 1
    return edges

def init():
    #musics = [os.path.splitext(f)[0] for f in os.listdir(folder)]
    folder = '/home/hparmantier/Montreux Analytics/Git/MarkovJazzFestival/Data/'
    musics = ['wall', 'billiejean']
    for i, music in enumerate(musics):
        print("########### "+music+" ############")
        print("########### "+str(i+1)+"/"+str(len(musics))+" #########")
        intra = builder.IntraGraph(folder+music+'.mp3')
        path = intra.simulate(save=True, filename=music+'.wav')
        edge_path = nodes2edges(path)
        d = {"array": edge_path}##
        d["beat_duration"] = intra.beat_duration##
        save2json(d,music)##
        neo = writer.connect_to_db()
        print("Pushing to Neo4j...")
        writer.intra_neo_from_nx(intra.G, edge_path, neo, music)
        print("Done")


def save2json(arr, f):
    folder = '/home/hparmantier/Montreux Analytics/Git/MarkovJazzFestival/Visualization/'
    out = folder+f+'_path.json'
    js = json.dumps(arr,out)
    f=open(out,'w')
    f.write(js)

def main():
    musics = ['billiejean', 'wall']
    for music in musics:
        folder = '/home/hparmantier/Montreux Analytics/Git/MarkovJazzFestival/Data/'
        intra = builder.IntraGraph(folder+music+'.mp3')
        path = intra.simulate(save=True, filename=music+'.wav')
        edge_path = nodes2edges(path)[::-1]
        d = {"array": edge_path}
        d["beat_duration"] = intra.beat_duration
        save2json(d,music)

def print_duration():
    folder = '/home/hparmantier/Montreux Analytics/Git/MarkovJazzFestival/Data/'
    musics = ['together']
    for music in musics:
        intra = builder.IntraGraph(folder+music+'.mp3')
        print(intra.beat_duration)
