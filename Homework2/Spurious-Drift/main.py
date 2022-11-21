#!/usr/bin/python

import numpy as np
import matplotlib
matplotlib.use("tkagg")

from matplotlib import pyplot as plt

BINS = 100
L = 100
LWR_BND = -L/2
UPR_BND = L/2

if __name__ == "__main__":

    for color, t in zip(["#537c78", "#cc222b", "#f15b4c", "#faa41b", "#ffd45b"],[1000]):
        with open(f"./output/{t}.csv") as f:
            contents = f.read()
            x = np.array(list(map(float, contents.split(",")[:-1])))

        counts, bins = np.histogram(x, bins=BINS, range=(LWR_BND,UPR_BND))
        print(f"  Avg:{np.mean(x[:])} Std:{np.std(x[:]) / L}")
        plt.stairs(counts,bins, fill=True, alpha=0.2, color=color, label=f"$t={t} s$")
        plt.stairs(counts,bins, color=color)


    plt.xlabel("$x_j$")
    plt.ylabel("Count")
    plt.legend()
    plt.grid()

    plt.show()
