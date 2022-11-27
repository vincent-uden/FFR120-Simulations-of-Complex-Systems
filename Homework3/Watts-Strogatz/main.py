#!/usr/bin/python

import numpy as np

def gen_matrix(n, c, p):
    A = np.zeros((n,n))
    for i in range(n):
        for j in range(i+1,n):
            # Periodic boundary
            dx = np.abs(i-j)
            if dx > n/2:
                dx = n - dx

            if  dx % n <= c/2:
                A[i,j] = 1
                A[j,i] = 1

            # Rewiring
            if np.random.rand() < p:
                A[i,j] = np.abs(A[i,j] - 1)
                A[j,i] = np.abs(A[j,i] - 1)
    return A


if __name__ == "__main__":
    A = gen_matrix(10, 4, 0.2)

    print(A)
