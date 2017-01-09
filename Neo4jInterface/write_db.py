from py2neo import Graph
from py2neo import Node, Relationship
from py2neo import neo4j
import pickle
import networkx as nx

def connect_to_db ():
    #usr = input("Enter username:")
    #pwd = input("Enter password:")
    return Graph("http://neo4j:parmNEO01@localhost:7474/db/data/")
    #remote_graph = Graph("http://localhost:7474/db/data") #connect to other address


def load_nx(file):
    return nx.read_gpickle(file)

def push_rel(n1,n2,w,neo):
    e = Relationship(n1, "Similar to", n2)
    e.properties["weight"] = w
    neo.create(e)

def create_node(graph_db, index, music):
    print(index)
    cypher = graph_db.cypher
    cypher.execute("CREATE (b:Beat{name:toInt({i}), music:{m}})", i=index, m=music)#index -> name pour affichage db

def create_rel(graph_db,music, u,v,w, in_path):
    cypher = graph_db.cypher
    visited = '0' if (in_path=="False") else '1'
    statement = "MATCH (u1:Beat {name:toInt({i1}), music:{m}}), (u2:Beat {name:toInt({i2}), music:{m}}) CREATE (u1)-[:SIMILARTO{similarity: toFloat({w}), visited:{visited}}]->(u2)"
    cypher.execute(statement, i1=u, i2=v, m=music, w=w, visited=visited)


def intra_neo_from_nx(nx,edge_path,graph_db, music):
    print('########### Nodes creations #############')
    for n in nx : create_node(graph_db, str(n), music) ## str(n) -> n

    print('###### Edges/Relationships creations ######')
    for u,v,pr in nx.edges(data=True):
        weight = pr['weight']
        #in_path = (pr['in_path'] == 1)
        next = 0 if (v == len(nx)-1) else v+1
        inp=in_path(u,v,edge_path)
        print("("+str(u)+","+str(next)+") similarity := "+str(weight) + ', in_path='+str(inp))
        create_rel(graph_db,music, str(u),str(next),str(weight), inp) ## str(u), str(v) -> u,v

def in_path(u,v, edge_path):
    has_u = list(filter(lambda e: e[0]==u , edge_path))
    for e1,e2 in has_u:
        if (e2==v):
            return True
    return False
