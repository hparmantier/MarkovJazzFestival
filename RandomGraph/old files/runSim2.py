import os
import sys
helper = '/home/hparmantier/Montreux Analytics/Scripts/Neo4jInterface'
sys.path.append(os.path.abspath(helper))
import SimulateRW as sim
import networkx as nx
import librosa

file = '../Data/wknd.gpickle'
G = nx.read_gpickle(file)
path = sim.generate_permutation_nx(G)

song = '../Data/wknd.mp3'
y, sr = librosa.load(song)
y_perm = sim.play_path(song, path)
librosa.output.write_wav('./wknd_test.wav', y_perm, sr)

