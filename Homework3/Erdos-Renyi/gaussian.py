import numpy as np

from main import *
from matplotlib import pyplot as plt

def normal_dist(x, mu=0.0, sigma=1.0):
    return 1/(sigma*np.sqrt(2*np.pi)) * np.exp(-0.5*((x-mu)/sigma)**2)

if __name__ == "__main__":
    for (i, N) in enumerate([100, 1000, 10000]):
        A = gen_graph(N, 0.01)

        x = np.linspace(-10, 10, num=N)

        (k, k_count, connections) = get_degrees(A, return_connetions=True)
        plt.subplot(1,3,i+1)
        plt.plot(k, normal_dist(k, mu=np.mean(connections), sigma=np.std(connections)), label="Gaussian")
        plt.plot(k, k_count/N, label="Data points")
        plt.title(f"$N = {N}$")

    plt.legend()
    plt.show()
