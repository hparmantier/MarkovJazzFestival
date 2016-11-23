from numpy.random import choice
import networkx as nx

p_file = '/home/hparmantier/Montreux Analytics/Media/distr_choice.txt'
f = open(p_file, 'w')

##TODO remove write in file
##IMPLEMENTATION MADE FOR LINKS U->U AND NOT TEMPORAL U->U+1
def make_step(G, current):
    print(current)
    edges = list(filter(lambda e: e[0]==current, G.edges(current, data=True)))
    #print(list(edges))
    neighbors = [e[1] for e in edges]
    pr = [e[2]['weight'] for e in edges]
    print("current:")
    print(current)
    print("neighbors:")
    print(neighbors)
    print("distribution:")
    print(pr)
    f.write("current:\n")
    f.write(str(current)+"\n")
    f.write("neighbors:\n")
    f.write(','.join(map(lambda i: str(i),neighbors))+"\n")
    f.write("distribution:\n")
    f.write(','.join(map(lambda i: str(i),pr))+"\n")
    last_node = len(G)-1
    str_ngb = list(map(lambda n: str(n), neighbors))
    chosen = int(choice(str_ngb, 1, pr)[0])
    #chosen = choice(neighbors, 1, pr)
    print("chosen:")
    print(chosen)
    f.write("chosen:\n")
    f.write(str(chosen)+"\n")

    #toint = int(chosen[0])
    #print(toint)
    return 0 if chosen == last_node else  int(chosen + 1)


def make_n_step(G, current, n):
    path = []
    curr = current
    for i in range(n):
        nxt = make_step(G, curr)
        print("next:")
        print(nxt)
        path.append(nxt)
        curr = nxt
    f.close()                                       ##to remove
    return path
