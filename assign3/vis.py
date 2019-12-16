import networkx as nx


def plot_network(graph, classes_disp, name='Chow-Liu'):
    G = nx.Graph()
    V, E = [], []
    V = {}
    for k, v in graph.items():
        V[k] = True
        for i in v:
            V[i] = True
            if len(v) == 0: continue
            E.append((k, i))

    labels = dict()
    for i, _ in V.items(): 
        labels[i] = classes_disp[i]

    for v, _ in V.items():
        G.add_node(labels[v])

    for (u,v) in E:
        G.add_edge(labels[u], labels[v])

    Gname = name + ".graphml"
    nx.write_graphml(G, Gname)




