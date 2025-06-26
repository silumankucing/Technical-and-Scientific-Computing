import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

# Grid size
nx, ny = 30, 30
N = nx * ny

# Create sparse matrix and right-hand side
A = lil_matrix((N, N))
b = np.zeros(N)

# Boundary conditions: top edge = 100, others = 0
for i in range(nx):
    for j in range(ny):
        idx = i * ny + j
        if i == 0 or i == nx-1 or j == 0:  # left, right, bottom
            A[idx, idx] = 1
            b[idx] = 0
        elif j == ny-1:  # top
            A[idx, idx] = 1
            b[idx] = 100
        else:
            A[idx, idx] = -4
            A[idx, idx-1] = 1
            A[idx, idx+1] = 1
            A[idx, idx-ny] = 1
            A[idx, idx+ny] = 1

# Solve the linear system
u = spsolve(A.tocsr(), b)
u_grid = u.reshape((nx, ny))

# Visualize the result
plt.imshow(u_grid.T, origin='lower', cmap='hot', extent=[0, 1, 0, 1])
plt.colorbar(label='Temperature')
plt.title('2D Finite Element Simulation (Laplace)')
plt.xlabel('x')
plt.ylabel('y')
plt.show()