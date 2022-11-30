#!/usr/bin/python

import numpy as np
import networkx
import scipy

from matplotlib import pyplot as plt

def gen_graph(n, p):
    A = np.zeros((n,n))
    for i in range(n):
        for j in range(i+1,n):
            A[i,j] = np.random.rand()<p
            A[j,i] = A[i,j]

    return A

def degree_prob(n, k, p):
    return scipy.special.comb(n-1, k) * np.power(p, k) * np.power((1 - p),n-1-k)

def get_degrees(A, return_connetions=False):
    n = A.shape[0]
    connections = np.zeros(n)
    for j in range(n):
        connections[j] = np.count_nonzero(A[j,:])
    if return_connetions:
        (i,x) = np.unique(connections, return_counts=True)
        return (i, x, connections)
    else:
        return np.unique(connections, return_counts=True)

if __name__ == "__main__":
    for (i, n, p) in [(1, 100, 0.05), (2, 400, 0.01), (3, 200, 0.05)]:
        A = gen_graph(n, p)

        G = networkx.from_numpy_array(A)

        plt.subplot(2, 3, i)
        plt.title(f"$n$: {n}, $p$: {p}")
        pos = networkx.circular_layout(G)
        networkx.draw(G, pos=pos, node_size=30)

        plt.subplot(2, 3, 3+i)
        plt.title(f"$n$: {n}, $p$: {p}")

        k, k_count = get_degrees(A)
        plt.bar(k, k_count / n)
        plt.plot(np.arange(np.max(k)), degree_prob(n, np.arange(np.max(k)), p), color="#E66C00")

    plt.show()
