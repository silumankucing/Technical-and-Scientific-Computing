import numpy as np
import matplotlib.pyplot as plt

# Define mesh size and domain

x = np.linspace(0, 1, 10)  # 10 points in x-direction
y = np.linspace(0, 1, 8)   # 8 points in y-direction

# Generate mesh grid

X, Y = np.meshgrid(x, y)

# visualize the mesh

plt.plot(X, Y, marker='o', color='k', linestyle='none')
plt.title('2D Mesh Grid')
plt.xlabel('x')
plt.ylabel('y')
plt.show()