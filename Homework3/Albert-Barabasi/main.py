import networkx as nx
import numpy as np

from matplotlib import pyplot as plt

from random import sample

def preferentialgrowth_graph(n,n0,m):
    A = np.zeros((n,n))
    for i in range(n0):
        for j in range(i+1,n0):
            A[i,j] = 1
    A = A + np.transpose(A)
    for t in range(n-n0):
        D = np.sum(A,axis=0)/np.sum(A)
        edges = np.random.choice(np.arange(n), m ,replace=False, p=D)
        for i in range(m):
            A[t+n0,edges[i]] = 1
            A[edges[i],t+n0] = 1

    return(A)

if __name__ == "__main__":
    for i, n, n0, m in [(1, 100, 5, 3), (2, 100, 15, 3), (3, 100, 10, 8)]:
        A = preferentialgrowth_graph(n, n0, m)

        G = nx.from_numpy_array(A)

        plt.subplot(1, 3, i)
        plt.title(f"$n = {n}, n_0 = {n0}, m = {m}$")
        pos = nx.circular_layout(G)
        nx.draw(G, pos=pos, node_size=30)

    plt.show()
