"""
Solusi numerik persamaan Laplace 1D menggunakan metode iteratif sederhana.
Persamaan: d2u/dx2 = 0
"""
import numpy as np
import matplotlib.pyplot as plt

def solve_laplace_1d(length=1.0, nx=50, tol=1e-6, max_iter=10000):
    dx = length / (nx - 1)
    x = np.linspace(0, length, nx)
    u = np.zeros(nx)
    u[0] = 1.0  # Boundary kiri
    u[-1] = 0.0  # Boundary kanan
    u_new = np.zeros(nx)

    for it in range(max_iter):
        for i in range(1, nx-1):
            u_new[i] = 0.5 * (u[i+1] + u[i-1])
        if np.linalg.norm(u_new - u, ord=np.inf) < tol:
            break
        u[:] = u_new[:]

    return x, u

if __name__ == "__main__":
    x, u = solve_laplace_1d()
    plt.plot(x, u)
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.title('Solusi Persamaan Laplace 1D')
    plt.show()
