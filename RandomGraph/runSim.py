import os
import sys
helper = '/home/hparmantier/Montreux Analytics/Scripts/Neo4jInterface'
sys.path.append(os.path.abspath(helper))
import write_db as writer
import read_db as reader
import SimulateRW as simulator

file = '/home/hparmantier/Montreux Analytics/Scripts/Data/creep_matrix.gpickle'
nx = writer.load_nx(file)
p_file = '/home/hparmantier/Montreux Analytics/Media/path.txt'

# for e in nx.edges(data=True):
#     print(e[2]['weight'])

#reader.cool_draw(nx) #-> draw for testing purpose


path = simulator.generate_permutation_nx(nx)
print(path)
print("Path Generated")
simulator.print_path(path, p_file)
print("Path printed")

beat_nb = len(nx)
music = "/home/hparmantier/Montreux Analytics/Scripts/Data/creep.mp3"
audio = simulator.play_permutation(music, path, beat_nb)
print(audio)
