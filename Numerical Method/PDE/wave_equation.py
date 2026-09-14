"""
Solusi numerik persamaan gelombang 1D menggunakan metode beda hingga eksplisit.
Persamaan: d2u/dt2 = c^2 * d2u/dx2
"""
import numpy as np
import matplotlib.pyplot as plt

def solve_wave_equation_1d(length=1.0, time=1.0, nx=50, nt=500, c=1.0):
    dx = length / (nx - 1)
    dt = time / nt
    x = np.linspace(0, length, nx)
    u = np.zeros(nx)
    u_new = np.zeros(nx)
    u_old = np.zeros(nx)

    # Kondisi awal: gelombang lonjakan di tengah
    u[int(nx/2)] = 1.0
    u_old[:] = u[:]

    for n in range(nt):
        for i in range(1, nx-1):
            u_new[i] = (2*u[i] - u_old[i] + (c*dt/dx)**2 * (u[i+1] - 2*u[i] + u[i-1]))
        u_old[:] = u[:]
        u[:] = u_new[:]

    return x, u

if __name__ == "__main__":
    x, u = solve_wave_equation_1d()
    plt.plot(x, u)
    plt.xlabel('x')
    plt.ylabel('Amplitude')
    plt.title('Solusi Persamaan Gelombang 1D')
    plt.show()
