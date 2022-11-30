#!/usr/bin/python

import numpy as np

import networkx as nx
import random


from matplotlib import pyplot as plt

def gen_matrix(n, c, p):
    A = np.zeros((n,n))
    for i in range(n):
        for j in range(int(c/2)):
            A[i,(i+j+1)%n] = 1
            if np.random.rand()<p:
                rewire_index = random.sample(list(np.where(np.logical_not(A[i,:])&np.logical_not(range(n)==i))[0]),1)
                A[i,(i+j+1)%n] = 0
                A[i,rewire_index] = 1
    A = A + np.transpose(A)
    return(A)


if __name__ == "__main__":

    for i, (c, p) in enumerate([(2, 0.0), (4, 0.0), (8, 0.0), (2, 0.2), (4, 0.2), (8, 0.2)]):
        A = gen_matrix(20, c, p)

        plt.subplot(2,3,i+1)
        plt.title(f"$c = {c}$")
        G = nx.from_numpy_array(A)
        pos = nx.circular_layout(G)
        nx.draw(G, pos, node_size=30)

    plt.show()

