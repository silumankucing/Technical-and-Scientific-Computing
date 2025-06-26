import numpy as np
import matplotlib.pyplot as plt

# Define mesh size and domain

x = np.linspace(0, 1, 6)   # 6 points in x-direction
y = np.linspace(0, 1, 5)   # 5 points in y-direction
z = np.linspace(0, 1, 4)   # 4 points in z-direction

# Generate mesh grid

X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# Visualize the mesh points in 3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X, Y, Z, color='k', marker='o')
ax.set_title('3D Mesh Grid')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()