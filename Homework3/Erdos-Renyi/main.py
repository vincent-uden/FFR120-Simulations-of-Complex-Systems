#!/usr/bin/python

import numpy as np
import networkx

from matplotlib import pyplot as plt

def gen_graph(n: int, p: float):
    R = np.random.random_sample((n,n)) < p
    # This symmetrizes the matrix and keeps the 50/50 random distribtuion since
    # XOR has 2 False outputs and 2 True outputs. It also has the added benefit
    # of setting the diagonal to 0
    R = np.logical_xor(R, R.T) * 1.0
    return R

if __name__ == "__main__":
    for (i, n, p) in [(1, 100, 0.01), (2, 10, 0.5), (3, 200, 0.05)]:
        A = gen_graph(n, p)

        print(A)
        G = networkx.from_numpy_array(A)

        plt.subplot(1, 3, i)
        plt.title(f"$n$: {n}, $p$: {p}")
        pos = networkx.circular_layout(G)
        networkx.draw(G, pos=pos, node_size=30)

    plt.show()
