import write_db as writer
import read_db as reader


file = '/home/hparmantier/Montreux Analytics/Project_Git/MarkovJazzFestival/Data/creep_mat.gpickle'
nx = writer.load_nx(file)
graph_db = writer.connect_to_db()
writer.intra_neo_from_nx(nx,graph_db)
