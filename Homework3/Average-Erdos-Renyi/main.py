#!/usr/bin/python

import numpy as np
import networkx as nx
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
    n = 500

    avg_length = np.zeros(100)
    cluster = np.zeros(100)
    for (i, p) in enumerate(np.arange(0,100) / 100.0):
        print(i)
        A = gen_graph(n, p)
        A_triple = (A @ A) @ A

        degrees = np.sum(A, axis=0)

        cluster[i] = np.sum(np.trace(A_triple)) / np.sum(degrees * (degrees - 1))

    print(cluster)
    plt.title(f"$n$: {n}")
    plt.plot(np.arange(100) / 100.0, cluster)
    plt.show()
