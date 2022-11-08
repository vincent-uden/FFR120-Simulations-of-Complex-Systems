#!/usr/bin/python

import numpy as np
import sys

sys.path.insert(0, "../Harmonic-Oscillator/")

from matplotlib import pyplot as plt
from integrate import leapfrog
from typing import Tuple
from tqdm import trange
from sklearn.preprocessing import normalize

def init_positions(size: Tuple[int, int], L: float, sigma: float):
    positions = np.zeros(size)

    for i in range(size[0]):
        point_placed = False
        print(i)

        while not point_placed:
            new_pos = np.random.rand(1, 2) * L

            distances = [ np.sqrt(np.sum((new_pos - positions[j,:])**2)) for j in range(i) ]
            too_close = [ r > sigma for r in distances ]

            if all(too_close):
                positions[i,:] = new_pos
                point_placed = True

    return positions

def init_velocities(size: Tuple[int, int], v0: float):
    angles = (np.random.rand(size[0]) * 2 * np.pi).reshape(size[0],1)
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    return directions * v0


def lennard_jones_force(positions: np.ndarray, epsilon: float, sigma: float) -> np.ndarray:
    forces = np.zeros_like(positions)

    for i in range(positions.shape[0]):
        # "Triangular" iteration avoids calculating force on self and duplicate calculations
        for j in range(i+1, positions.shape[0]):
            r = np.sqrt(np.sum((positions[i,:] - positions[j,:])**2))
            magnitude = 4 * epsilon * ( 12*np.power(sigma,12)*np.power(r, -13) - 6*np.power(sigma,6)*np.power(r,-7) )
            # Direction of force exerted on i (towards j)
            direction = normalize((positions[j,:] - positions[i,:]).reshape(1,-1)).squeeze()

            forces[i,:] -= magnitude*direction
            forces[j,:] += magnitude*direction

    return forces

# TODO: Implement boundary reflections

if __name__ == "__main__":
    m       = 1.0
    epsilon = 1.0
    sigma   = 1.0

    velocity_scale = np.sqrt(2*epsilon/m)
    time_scale = sigma*np.sqrt(m/(2*epsilon))

    L = sigma * 2
    N = 5

    positions = init_positions((N,2), L, sigma)
    #positions = np.array([[-sigma, 0.0], [sigma, 0.0]])
    velocities = init_velocities((N,2), 2*velocity_scale*0.1)
    #velocities = np.array([[0.0, 0.1], [0.0, -0.1]])

    print(positions.shape, velocities.shape)

    for t in trange(1000):
        if t % 10 == 0:
            plt.plot(positions[:,0], positions[:,1], '.')

        (positions, velocities) = leapfrog(positions, velocities, lambda x: lennard_jones_force(x, epsilon, sigma), dt=sigma/(2*velocity_scale) * 0.05)

    #plt.gca().set_xlim([0, L])
    #plt.gca().set_ylim([0, L])
    plt.show()