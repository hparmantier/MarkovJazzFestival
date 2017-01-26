import os
import sys
import json
helper = '/home/hparmantier/Montreux Analytics/Git/MarkovJazzFestival/RandomGraph'
sys.path.append(os.path.abspath(helper))
import intra as builder
import inter as builder2
import write_db as writer
import read_db as reader
import networkx as nx
import SimulateRW as simulator
import pickle

################################################################################

def nodes2edges(nodes):
    edges = []
    i = 1
    while i != len(nodes)-1:
        edges.append((nodes[i-1], nodes[i]))
        i += 1
    return edges

def init():
    folder = '/home/hparmantier/Montreux Analytics/Git/MarkovJazzFestival/Data/'
    #musics = [os.path.splitext(f)[0] for f in os.listdir(folder)]
    musics = ['Jamiroquai-Cosmic Girl']
    for i, music in enumerate(musics):
        print("########### "+music+" ############")
        print("########### "+str(i+1)+"/"+str(len(musics))+" #########")
        intra = builder.IntraGraph(folder+music+'.mp3')
        path = intra.simulate(save=True, filename=music+'.wav')
        edge_path = nodes2edges(path)[::-1]
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

def check_duration():
    folder = '/home/hparmantier/Montreux Analytics/Data/musics/inter/'
    musics = [os.path.splitext(f)[0] for f in os.listdir(folder)]
    for music in musics:
        intra = builder.IntraGraph(folder+music+'.mp3')
        print(music+': '+str(intra.beat_duration))

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


def interintra():
    folder = '/home/hparmantier/Montreux Analytics/Git/MarkovJazzFestival/Data/'
    files = ['00', '01']
    for f in files:
        path_handler = open(folder+'test'+f+'.pkl', 'rb')
        path = pickle.load(path_handler)

        inter_handler = open(folder+'int'+f+'.pkl', 'rb')
        inter = pickle.load(inter_handler)

        edge_path = nodes2edges(path)[::-1]
        d = {"array": edge_path}
        d["beat_duration"] = inter.songs[0].beat_duration
        save2json(d,'inter'+f)

        neo = writer.connect_to_db()
        print("Pushing to Neo4j...")
        writer.intra_neo_from_nx(inter.im, edge_path, neo, 'inter'+f)
        print("Done")



interintra()






def print_duration():
    folder = '/home/hparmantier/Montreux Analytics/Git/MarkovJazzFestival/Data/'
    musics = ['together']
    for music in musics:
        intra = builder.IntraGraph(folder+music+'.mp3')
        print(intra.beat_duration)
