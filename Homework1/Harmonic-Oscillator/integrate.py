import numpy as np

from typing import Callable, Tuple

def euler_step(pos: np.ndarray, vel: np.ndarray, acc: np.ndarray, dt = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    return (pos + vel*dt, vel + acc*dt)

def leapfrog(pos: np.ndarray, vel: np.ndarray, acc_func: Callable[[np.ndarray], np.ndarray], dt = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    pos_half = pos + vel * dt/2
    acc_half = acc_func(pos_half)

    vel_next = vel + acc_half * dt
    pos_next = pos_half + vel_next * dt/2
    return (pos_next, vel_next)
