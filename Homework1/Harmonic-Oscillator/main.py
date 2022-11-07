#!/usr/bin/python

# System libraries
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.ticker import StrMethodFormatter

# Other project files
from integrate import euler_step, leapfrog

EULER = 0
LEAPFROG = 1

# Assumes equilibrium at x = 0
def spring_force(pos: np.ndarray, k: float):
    return (-k) * pos

def analytical_positions(t: np.ndarray, A: float, omega: float, phi=0.0):
    return A * np.cos(omega*t+phi)

def analytical_velocities(t: np.ndarray, A: float, omega: float, phi=0.0):
    return -omega*A*np.sin(omega*t+phi)

def total_energy(pos: np.ndarray, vel: np.ndarray, k: float, m: float):
    return 0.5 * (k*(pos*pos) + m*(vel*vel))

if __name__ == "__main__":
    m = 0.1
    k = 5.0
    dts = [0.0001, 0.002, 0.02]

    params = {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Serif"],
        "font.size": 16,
        "figure.figsize": (14, 8)
    }
    plt.rcParams.update(params)

    # Set to EULER or LEAPFROG
    integrator = LEAPFROG

    T = 5 # Simulate for 10 seconds
    for plot_i in range(3):
        dt = dts[plot_i]
        time_steps = int(T / dt)

        position = np.array([0.1])
        velocity = np.array([0.0])
        acceleration = np.array([0.0])

        position_history = np.zeros((time_steps,1))
        energy_history = np.zeros((time_steps,1))

        for t in range(time_steps):
            position_history[t] = position
            energy_history[t] = total_energy(position, velocity, k, m)

            if integrator == EULER:
                acceleration = spring_force(position, k) / m
                (position, velocity) = euler_step(position, velocity, acceleration, dt=dt)
            elif integrator == LEAPFROG:
                (position, velocity) = leapfrog(position, velocity, lambda x: spring_force(x, k)/m, dt=dt)

        times = np.arange(time_steps) * dt
        analytical_sol = analytical_positions(times, 0.1, np.sqrt(k/m))
        analytical_v = analytical_velocities(times, 0.1, np.sqrt(k/m))
        analytical_energy = total_energy(
                analytical_sol,
                analytical_v,
                k,
                m
        )

        plt.subplot(2, 3, plot_i+1)
        plt.plot(times, position_history[:,0], ".", markersize=1)
        plt.plot(times, analytical_sol)
        plt.title(f"$\Delta t = {dt * 1000:.1f}$ ms")
        plt.ylabel("$x(t)$")

        plt.subplot(2, 3, plot_i+4)
        plt.plot(times, energy_history/analytical_energy[0]-1)
        plt.plot(times, analytical_energy/analytical_energy[0]-1)
        plt.xlabel("$t$")
        plt.ylabel("$(E(t)-E_0)/E_0$")

    plt.tight_layout()
    plt.show()
