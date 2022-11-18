#!/usr/bin/python

import numpy as np

import matplotlib
matplotlib.use("tkagg")

from matplotlib import pyplot as plt

def next_step(x, sigma, dt_sqrt):
    dirs = np.random.random_sample(N)
    diff = np.round(dirs) * 2 - 1
    return diff * sigma * dt_sqrt

if __name__ == "__main__":
    x0 = 0
    sigma = 1.0
    dt = 0.01
    dt_sqrt = np.sqrt(dt)

    N = 10000
    T = 101
    L = 100

    x = np.zeros(N)
    x[:] = x0

    t = 0.0
    j = 0

    BINS = 31
    LWR_BND = -L/2
    UPR_BND = L/2

    while t < T:
        diff = np.round(np.random.random_sample(N)) * 2 - 1
        x[:] = x[:] + next_step(x, sigma, dt_sqrt)

        # Reflection is VERY SLOW. Can we speed up?
        for n in range(N):
            if x[n] < -L/2.0:
                x[n] = -L - x[n]
            if x[n] > L/2.0:
                x[n] = L - x[n]

        if j == int(10 / dt) or j == int(100 / dt) or j == int(1000 / dt) or j == int(10000 / dt) or j == int(100000 / dt):
            counts, bins = np.histogram(x[:], bins=BINS, range=(LWR_BND,UPR_BND))
            print(f"j={j}")
            print(f"  Avg:{np.mean(x[:])} Std:{np.std(x[:]) / L}")
            plt.stairs(counts,bins, fill=True, alpha=0.2, color="#590995", label=f"$t={t}$")
            plt.stairs(counts,bins, color="#590995")

        t += dt
        j += 1


    plt.xlabel("$x_j$")
    plt.ylabel("Count")
    plt.legend()
    plt.grid()

    plt.show()
