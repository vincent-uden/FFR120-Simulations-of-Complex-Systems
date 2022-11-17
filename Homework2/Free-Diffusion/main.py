#!/usr/bin/python

import numpy as np

import matplotlib
matplotlib.use("tkagg")

from matplotlib import pyplot as plt

if __name__ == "__main__":
    x0 = 0
    sigma = 1.0
    dt = 1.0
    dt_sqrt = np.sqrt(dt)

    N = 10000
    T = 1000

    x = np.zeros((N,int(T/dt)))
    x[:,0] = x0

    t = 0.0
    j = 0

    while t < T-1:
        diff = np.round(np.random.random_sample((1,N))) * 2 - 1
        x[:,j+1] = x[:,j] + diff * sigma * dt_sqrt

        t += dt
        j += 1

    BINS = 31
    LWR_BND = -j*sigma*dt_sqrt / 10
    UPR_BND = j*sigma*dt_sqrt / 10
    counts0, bins0 = np.histogram(x[:,j//10], bins=BINS, range=(LWR_BND,UPR_BND))
    counts1, bins1 = np.histogram(x[:,j//2], bins=BINS, range=(LWR_BND,UPR_BND))
    counts2, bins2 = np.histogram(x[:,j], bins=BINS, range=(LWR_BND,UPR_BND))

    print(f"j={j//10}")
    print(f"  Avg:{np.mean(x[:,j//10])} Std:{np.std(x[:,j//10])} Expected std:{sigma*np.sqrt(2*j//10*dt)}")
    print(f"j={j//2}")
    print(f"  Avg:{np.mean(x[:,j//2])} Std:{np.std(x[:,j//2])} Expected std:{sigma*np.sqrt(2*j//2*dt)}")
    print(f"j={j}")
    print(f"  Avg:{np.mean(x[:,j])} Std:{np.std(x[:,j])} Expected std:{sigma*np.sqrt(2*j*dt)}")

    plt.stairs(counts0,bins0, fill=True, alpha=0.2, color="#590995", label=f"$j={j//10}$")
    plt.stairs(counts1,bins1, fill=True, alpha=0.2, color="#03C4A1", label=f"$j={j//2}$")
    plt.stairs(counts2,bins2, fill=True, alpha=0.2, color="#C62A88", label=f"$j={j}$")
    plt.stairs(counts0,bins0, color="#590995")
    plt.stairs(counts1,bins1, color="#03C4A1")
    plt.stairs(counts2,bins2, color="#C62A88")

    plt.xlabel("$x_j$")
    plt.ylabel("Count")
    plt.legend()
    plt.grid()

    plt.show()
