import numpy as np
import matplotlib.pyplot as plt
from stl import mesh

# Define mesh size and domain
x = np.linspace(0, 1, 10)
y = np.linspace(0, 1, 8)
X, Y = np.meshgrid(x, y)

# Create vertices for the mesh (z=0 for 2D)
vertices = np.column_stack((X.ravel(), Y.ravel(), np.zeros(X.size)))

# Create faces (two triangles per grid square)
faces = []
num_x = len(x)
num_y = len(y)
for i in range(num_y - 1):
    for j in range(num_x - 1):
        idx = i * num_x + j
        # Triangle 1
        faces.append([idx, idx + 1, idx + num_x])
        # Triangle 2
        faces.append([idx + 1, idx + num_x + 1, idx + num_x])

faces = np.array(faces)

# Create the mesh
surface = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
for i, f in enumerate(faces):
    for j in range(3):
        surface.vectors[i][j] = vertices[f[j], :]

# Save to STL file
surface.save('2d_mesh.stl')

# Optional: visualize the mesh
plt.plot(X, Y, marker='o', color='k', linestyle='none')
plt.title('2D Mesh Grid')
plt.xlabel('x')
plt.ylabel('y')
plt.show()