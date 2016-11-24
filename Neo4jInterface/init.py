import write_db as writer
import read_db as reader


file = '/home/hparmantier/Montreux Analytics/Git/MarkovJazzFestival/Data/creep_directed.gpickle'
nx = writer.load_nx(file)
# for e in nx.edges(data=True):
#     print(e)
# print(len(nx.edges(data=True)))
reader.cool_draw(nx)
graph_db = writer.connect_to_db()
writer.intra_neo_from_nx(nx,graph_db)
