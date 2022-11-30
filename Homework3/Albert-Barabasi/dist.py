import networkx as nx
import numpy as np

from matplotlib import pyplot as plt

from main import preferentialgrowth_graph

if __name__ == "__main__":
    n  = 1000
    m  = 3
    n0 = 5

    A = preferentialgrowth_graph(n, n0, m)
    degrees = np.sum(A,1)

    plt.loglog(np.sort(degrees)[::-1],np.arange(n)/n,'.')
    plt.loglog(np.arange(m,np.max(degrees)), m**2 * np.arange(m,np.max(degrees))**(-2),'--')

    plt.title("Degree distribution")
    plt.ylabel('$C(k)$')
    plt.xlabel('$k$')

    plt.show()
