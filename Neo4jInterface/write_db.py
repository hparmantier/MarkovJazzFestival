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
    cypher.execute("CREATE (b:Beat{name:{i}, music:{m}})", i=index, m=music)#index -> name pour affichage db

def create_rel(graph_db,u,v,w):
    cypher = graph_db.cypher
    statement = "MATCH (u1:Beat {name:toInt({i1}), music:{m}}), (u2:Beat {name:toInt({i2}), music:{m}}) CREATE (u1)-[:SIMILARTO{similarity: toFloat({w})}]->(u2)"
    cypher.execute(statement, i1=u, i2=v, m="Creep", w=w) ## add weight in string format which then converted to float in db


def intra_neo_from_nx(nx,graph_db):
    print('###########Nodes creations #############')
    for n in nx : create_node(graph_db, n, "Creep") ## str(n) -> n

    print('###### Edges/Relationships creations ######')
    for u,nbrsdict in nx.adjacency_iter():
        for v,eattr in nbrsdict.items():
            if 'weight' in eattr:
                weight = eattr['weight']
                print("("+str(u)+","+str(v)+") similarity := "+str(weight))
                create_rel(graph_db,str(u),str(v),str(weight)) ## str(u), str(v) -> u,v
