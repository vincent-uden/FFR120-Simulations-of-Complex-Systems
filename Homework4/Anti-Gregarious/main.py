
import numpy as np

from matplotlib import pyplot as plt
from scipy.signal import fftconvolve
from tqdm import trange

IMAGE = 0
HAPPINESS = 1

def get_majorities(town: np.ndarray) -> int:
    kernel = np.ones((3, 3))
    conv = fftconvolve(town, kernel, mode="same")
    # Has the same value as the majority (or 0) of the neighbourghood
    major = (conv > 0.0) * 1 - (conv < 0.0) * 1
    return major

if __name__ == "__main__":
    N = 50
    town = np.concatenate((np.zeros(int(N*N*0.1)), np.ones(int(N*N*0.45)), np.ones(int(N*N*0.45)) * (-1)))
    np.random.shuffle(town)
    town = town.reshape((N,N))

    town0 = np.copy(town)


    T = 100000
    a_happy = np.zeros(T)
    b_happy = np.zeros(T)
    moving  = np.zeros(T)

    plotting = HAPPINESS

    for t in trange(T):
        majorities = get_majorities(town)
        happy = majorities * town

        a_happy[t] = np.count_nonzero(np.logical_and(happy < 0, town > 0)) / int(N*N*0.45)
        b_happy[t] = np.count_nonzero(np.logical_and(happy < 0, town < 0)) / int(N*N*0.45)
        # Find non-empty house
        while True:
            i = np.random.randint(0, N)
            j = np.random.randint(0, N)
            if town[i,j] != 0:
                break

        if happy[i, j] == 1:
            # Find empty house to move to
            moving[t] = 1
            while True:
                i_move = np.random.randint(0, N)
                j_move = np.random.randint(0, N)
                if town[i_move,j_move] == 0 and i_move != i and j_move != j:
                    break

            town[i_move,j_move] = town[i,j]
            town[i,j] = 0

    if plotting == IMAGE:
        plt.subplot(1, 2, 1)
        plt.imshow(town0, cmap="bwr")
        plt.title("$t=1$")

        plt.subplot(1, 2, 2)
        plt.imshow(town, cmap="bwr")
        plt.title(f"$t={T}$")
    elif plotting == HAPPINESS:
        plt.plot(np.arange(T), a_happy, label="Happiness (A)")
        plt.plot(np.arange(T), b_happy, label="Happiness (B)")
        plt.plot(np.arange(T), (a_happy + b_happy) / 2, label="Happiness (total)")
        plt.plot(np.arange(T)[1000:], (fftconvolve(moving, np.ones(1000), mode="same")/1000)[1000:], label="Moves (rolling mean, window=1000)")
        plt.legend()
        plt.xlabel("$t$")
        plt.ylabel("$p$")

    plt.show()


