import edit_db as edb

file = '/home/hparmantier/Montreux Analytics/Scripts/data/creep_graph.gpickle'
nx = edb.load_nx(file)
graph_db = edb.connect_to_db()
edb.intra_neo_from_nx(nx,graph_db)
