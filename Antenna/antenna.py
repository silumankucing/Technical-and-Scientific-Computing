import numpy as np
import matplotlib.pyplot as plt

# Grid size and parameters

nx, ny = 100, 100
V = np.zeros((nx, ny))

# Place a point charge in the center

charge_x, charge_y = nx // 2, ny // 2
rho = np.zeros((nx, ny))
rho[charge_x, charge_y] = 1e3

# Parameters

epsilon = 8.85e-12
dx = 1.0

# Iterative solver (Gauss-Seidel)

for it in range(5000):
    V[1:-1,1:-1] = 0.25 * (V[2:,1:-1] + V[:-2,1:-1] + V[1:-1,2:] + V[1:-1,:-2] + dx**2 * rho[1:-1,1:-1]/epsilon)

# Dirichlet boundary: V=0 at edges

# Visualize the potential

plt.imshow(V.T, origin='lower', cmap='plasma')
plt.colorbar(label='Potential (V)')
plt.title('2D Electrostatics Simulation (Potential Field)')
plt.xlabel('x')
plt.ylabel('y')
plt.show()