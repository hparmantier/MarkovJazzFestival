from numpy.random import choice
import networkx as nx

def make_step(G, current):
    edges = filter(lambda e: e[0]==current, G.edges(current, data=True))
    neighbors = [e[1] for e in edges]
    pr = [e[2]['weight'] for e in edges]
    print(current)
    print(neighbors)
    print(pr)
    last_node = len(G)-1
    chosen = choice(neighbors, 1, pr)
    return 0 if chosen == last_node else  int(chosen + 1) ##temporal neighbor of node chosen TODO: if last temporal node return 0 (first beat)

def make_n_step(G, current, n):
    path = []
    curr = current
    for i in range(n):
        nxt = make_step(G, curr)
        path.append(nxt)
        curr = nxt
    return path
