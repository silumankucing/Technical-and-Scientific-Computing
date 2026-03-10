"""
Solusi numerik persamaan Poisson 1D menggunakan metode Jacobi.
Persamaan: d2u/dx2 = f(x)
"""
import numpy as np
import matplotlib.pyplot as plt

def solve_poisson_1d(length=1.0, nx=50, tol=1e-6, max_iter=10000):
    dx = length / (nx - 1)
    x = np.linspace(0, length, nx)
    u = np.zeros(nx)
    f = np.sin(np.pi * x)  # Contoh sumber
    u_new = np.zeros(nx)

    for it in range(max_iter):
        for i in range(1, nx-1):
            u_new[i] = 0.5 * (u[i+1] + u[i-1] - dx**2 * f[i])
        if np.linalg.norm(u_new - u, ord=np.inf) < tol:
            break
        u[:] = u_new[:]

    return x, u

if __name__ == "__main__":
    x, u = solve_poisson_1d()
    plt.plot(x, u)
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.title('Solusi Persamaan Poisson 1D')
    plt.show()
