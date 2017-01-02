import os
import sys
helper = '/home/hparmantier/Montreux Analytics/Git/MarkovJazzFestival'
sys.path.append(os.path.abspath(helper))
import RandomGraph.graph as builder
import write_db as writer
import read_db as reader
import networkx as nx
import RandomGraph.SimulateRW as simulator

musics = ['opa', 'creep', 'gangnamstyle', 'mildhighclub']

def nodes2edges(nodes):
    edges = []
    i = 1
    while i != len(nodes)-1:
        edges.append((nodes[i-1], nodes[i]))
        i += 1
    return edges

def init2(folder, musics):
    for music in musics:
        print("########### "+music+" ############")
        song = folder+music+'.mp3'
        print("Building graph networkx...")
        graph_file = folder+'/graphs/graph_'+music+'.gpickle'
        G = builder.song2graph(song, graph_file)
        print("Done.")
        print("Simulate RandomWalk")
        sim_file = folder+'/simulations/sim_'+music+'.wav'
        builder.simulate(song, G, sim_file)
        print("Done")
        """
        print("Pushing graph to Neo4j...")
        edge_path = nodes2edges(path)
        neo = writer.connect_to_db()
        writer.intra_neo_from_nx(G,edge_path,neo, music)
        print("Done")
        """



def init1(file):
    nx = writer.load_nx(file)
    reader.cool_draw(nx)
    graph_db = writer.connect_to_db()
    writer.intra_neo_from_nx(nx,graph_db)
