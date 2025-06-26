import numpy as np
from stl import mesh

x = np.linspace(0, 1, 6)   # 6 points in x-direction
y = np.linspace(0, 1, 5)   # 5 points in y-direction
z = np.linspace(0, 1, 4)   # 4 points in z-direction

# Generate mesh grid
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# Flatten the grid points to vertices
vertices = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))

# Create hexahedral cells and convert to triangular faces
faces = []
nx, ny, nz = len(x), len(y), len(z)
for i in range(nx - 1):
    for j in range(ny - 1):
        for k in range(nz - 1):
            # 8 corners of the hexahedron
            n0 = i * ny * nz + j * nz + k
            n1 = (i + 1) * ny * nz + j * nz + k
            n2 = (i + 1) * ny * nz + (j + 1) * nz + k
            n3 = i * ny * nz + (j + 1) * nz + k
            n4 = i * ny * nz + j * nz + (k + 1)
            n5 = (i + 1) * ny * nz + j * nz + (k + 1)
            n6 = (i + 1) * ny * nz + (j + 1) * nz + (k + 1)
            n7 = i * ny * nz + (j + 1) * nz + (k + 1)
            # Each hexahedron has 6 faces, each face is 2 triangles
            # Bottom face (n0, n1, n2, n3)
            faces.append([n0, n1, n2])
            faces.append([n0, n2, n3])
            # Top face (n4, n5, n6, n7)
            faces.append([n4, n5, n6])
            faces.append([n4, n6, n7])
            # Front face (n0, n1, n5, n4)
            faces.append([n0, n1, n5])
            faces.append([n0, n5, n4])
            # Back face (n3, n2, n6, n7)
            faces.append([n3, n2, n6])
            faces.append([n3, n6, n7])
            # Left face (n0, n3, n7, n4)
            faces.append([n0, n3, n7])
            faces.append([n0, n7, n4])
            # Right face (n1, n2, n6, n5)
            faces.append([n1, n2, n6])
            faces.append([n1, n6, n5])

faces = np.array(faces)

# Create the mesh

cube_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
for i, f in enumerate(faces):
    for j in range(3):
        cube_mesh.vectors[i][j] = vertices[f[j], :]

# Save to STL file

cube_mesh.save('mesh.stl')