#!/usr/bin/python

import numpy as np
import sys

sys.path.insert(0, "../Harmonic-Oscillator/")

from matplotlib import pyplot as plt
from integrate import leapfrog
from typing import Tuple

def init_positions(size: Tuple[int, int], L: float, sigma: float):
    return

def init_positions(size: Tuple[int, int], L: float, v0: float):
    return

def lennard_jones_force(positions: np.ndarray, epsilon: float, sigma: float):
    return

if __name__ == "__main__":
    m       = 1.0
    epsilon = 1.0
    sigma   = 1.0

    velocity_scale = np.sqrt(2*epsilon/m)
    time_scale = sigma*np.sqrt(m/(2*epsilon))

    L = sigma * 100
    N = 100

    positions = np.zeros((N, 2))