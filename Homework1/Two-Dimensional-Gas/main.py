#!/usr/bin/python

import numpy as np
import os
import sys

sys.path.insert(0, "../Harmonic-Oscillator/")

from datetime import datetime
from matplotlib import pyplot as plt
from integrate import leapfrog
from typing import Tuple
from tqdm import trange

SNAPSHOT = 0
ENERGIES = 1

def init_positions(size: Tuple[int, int], L: float, sigma: float):
    positions = np.zeros(size)

    for i in range(size[0]):
        point_placed = False

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

def lennard_jones_potential(positions: np.ndarray, epsilon: float, sigma: float) -> np.ndarray:
    potentials = np.zeros(positions.shape[0])

    for i in range(positions.shape[0]):
        # "Triangular" iteration avoids calculating force on self and duplicate calculations
        for j in range(i+1, positions.shape[0]):
            r = (np.sum((positions[i,:] - positions[j,:])**2))
            magnitude = 4 * epsilon * ( np.power(sigma**2/r,6) - np.power(sigma**2/r,3) )

            potentials[i] += magnitude
            potentials[j] += magnitude
    return potentials

def lennard_jones_force(positions: np.ndarray, epsilon: float, sigma: float) -> np.ndarray:
    forces = np.zeros_like(positions)

    for i in range(positions.shape[0]):
        # "Triangular" iteration avoids calculating force on self and duplicate calculations
        for j in range(i+1, positions.shape[0]):
            r = np.sqrt(np.sum((positions[i,:] - positions[j,:])**2))
            magnitude = 4 * epsilon * ( 12*np.power(sigma,12)*np.power(r, -13) - 6*np.power(sigma,6)*np.power(r,-7) )
            # Direction of force exerted on i (towards j)
            direction = (positions[j,:] - positions[i,:]) / r

            forces[i,:] -= magnitude*direction
            forces[j,:] += magnitude*direction
    return forces

def kinetic_energy(velocities: np.ndarray, m: float, velocity_scale: float) -> float:
    return 0.5 * m * np.sum((velocities/velocity_scale)**2)

def potential_energy(positions: np.ndarray, epsilon: float, sigma: float) -> float:
    # Check the math on this one
    return 0.5 * np.sum(lennard_jones_potential(positions, epsilon, sigma))

if __name__ == "__main__":
    m       = 0.1
    epsilon = 1.0
    sigma   = 1.0

    velocity_scale = np.sqrt(2*epsilon/m)
    time_scale = sigma*np.sqrt(m/(2*epsilon))

    L = sigma * 100
    N = 10

    positions = init_positions((N,2), L, sigma)
    velocities = init_velocities((N,2), 2*velocity_scale)

    time_steps = 50000
    plot_freq = 5
    dt = sigma/(2*velocity_scale) * 0.02
    position_history = np.empty((time_steps//plot_freq,N,2))

    E_k_history = np.empty(time_steps//plot_freq)
    E_p_history = np.empty(time_steps//plot_freq)

    plotting = ENERGIES
    logging = False

    for t in trange(time_steps):
        if logging or plotting == SNAPSHOT:
            position_history[t // plot_freq,:,:] = positions
        if t % plot_freq == 0 and plotting == ENERGIES:
            # Something is wrong with the units. Graphs have the correct shape but are not of the same scale
            E_k_history[t//plot_freq] = kinetic_energy(velocities, m, velocity_scale)
            E_p_history[t//plot_freq] = potential_energy(positions, epsilon, sigma)

        (positions, velocities) = leapfrog(positions, velocities, lambda x: lennard_jones_force(x, epsilon, sigma), dt=dt)

        for i in range(N):
            # Outside left bound
            if positions[i,0] < 0:
                positions[i,0] = positions[i,0] * -1
                velocities[i,0] = velocities[i,0] * -1

            # Outside right bound
            elif positions[i,0] > L:
                positions[i,0] = 2*L - positions[i,0]
                velocities[i,0] = velocities[i,0] * -1

            # Outside lower bound
            elif positions[i,1] < 0:
                positions[i,1] = positions[i,1] * -1
                velocities[i,1] = velocities[i,1] * -1

            # Outside upper bound
            elif positions[i,1] > L:
                positions[i,1] = 2*L - positions[i,1]
                velocities[i,1] = velocities[i,1] * -1

    if plotting == SNAPSHOT:
        for i in range(N):
            plt.plot(position_history[:,i,0].squeeze(), position_history[:,i,1].squeeze())
        plt.gca().set_xlim([0, L])
        plt.gca().set_ylim([0, L])
    elif plotting == ENERGIES:
        plt.subplot(3, 1, 1)
        plt.plot(np.arange(time_steps//plot_freq) * dt / time_scale, E_k_history / epsilon, '', label="Kinetic Energy", markersize=1)
        plt.ylabel("$E_k$")
        plt.subplot(3, 1, 2)
        plt.plot(np.arange(time_steps//plot_freq) * dt / time_scale, E_p_history / epsilon, '', label="Potential Energy", markersize=1)
        plt.ylabel("$E_p$")
        plt.subplot(3, 1, 3)
        plt.plot(np.arange(time_steps//plot_freq) * dt / time_scale, (E_k_history + E_p_history / epsilon), '', label="Potential Energy", markersize=1)
        plt.ylabel("$E$")

    if logging:
        try:
            os.mkdir("./logs")
        except FileExistsError:
            pass
        np.savetxt("./logs/" + datetime.now().strftime("%d-%m-%Y-%H:%M:%S") + ".txt", position_history.reshape(position_history.shape[0],N*2))
    plt.show()
