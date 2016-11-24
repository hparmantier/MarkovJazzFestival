import os
import sys
helper = '/home/hparmantier/Montreux Analytics/Git/MarkovJazzFestival'
sys.path.append(os.path.abspath(helper))
import Neo4jInterface.write_db as conn
#import write_db as conn
import networkx as nx
import matplotlib.pyplot as plt

def intra_graph(music):
    neo = conn.connect_to_db()
    cypher = neo.cypher
    statement = "MATCH (n1:Beat{music:{m}})-[r:SIMILARTO]->(n2:Beat{music:{m}}) RETURN n1, r, n2"
    res = cypher.execute(statement, m=music)
    return neo_to_nx(res)
    #print(len(res))

def neo_to_nx(records):
    G = nx.Graph()
    for record in records:
        G.add_edge(int(record[0]['name']), int(record[2]['name']), weight=record[1]['similarity'])
    return G

def stream_built_nx(music):
    neo = conn.connect_to_db()
    G = nx.Graph()
    statement = "MATCH (n1:Beat{music:{m}})-[r:SIMILARTO]->(n2:Beat{music:{m}}) RETURN n1, r, n2"
    cypher = neo.cypher
    for record in neo.cypher.stream(statement, m=music):
        G.add_edge(int(record[0]['name']), int(record[2]['name']), weight=record[1]['similarity'])
    print(G.size())
    print(len(G))
    return G

def cool_draw(G):
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G,pos, node_color='blue', node_size=20, alpha=0.8)
    nx.draw_networkx_edges(G,pos,width=1.0,alpha=1, edge_color='k')
    plt.show()
