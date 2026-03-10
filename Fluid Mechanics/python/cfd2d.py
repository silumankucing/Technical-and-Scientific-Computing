import numpy as np
import matplotlib.pyplot as plt

# Parameter domain
nx, ny = 50, 50  # jumlah grid
dx, dy = 2.0 / (nx - 1), 2.0 / (ny - 1)
x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)

# Parameter simulasi
nt = 100  # jumlah langkah waktu
nu = 0.1  # viskositas kinematik
dt = 0.001  # langkah waktu

# Inisialisasi variabel
u_field = np.ones((ny, nx))
v_field = np.ones((ny, nx))

# Kondisi awal: gangguan di tengah domain
u_field[int(0.5 / dy):int(1 / dy + 1), int(0.5 / dx):int(1 / dx + 1)] = 2
v_field[int(0.5 / dy):int(1 / dy + 1), int(0.5 / dx):int(1 / dx + 1)] = 2

# Simulasi CFD 2D: Persamaan adveksi-difusi sederhana
for n in range(nt):
    un = u_field.copy()
    vn = v_field.copy()
    u_field[1:, 1:] = (un[1:, 1:] -
                      un[1:, 1:] * dt / dx * (un[1:, 1:] - un[1:, :-1]) -
                      vn[1:, 1:] * dt / dy * (un[1:, 1:] - un[:-1, 1:]) +
                      nu * dt / dx**2 * (un[1:, 2:] - 2 * un[1:, 1:] + un[1:, :-1]) +
                      nu * dt / dy**2 * (un[2:, 1:] - 2 * un[1:, 1:] + un[:-1, 1:]))
    v_field[1:, 1:] = (vn[1:, 1:] -
                      un[1:, 1:] * dt / dx * (vn[1:, 1:] - vn[1:, :-1]) -
                      vn[1:, 1:] * dt / dy * (vn[1:, 1:] - vn[:-1, 1:]) +
                      nu * dt / dx**2 * (vn[1:, 2:] - 2 * vn[1:, 1:] + vn[1:, :-1]) +
                      nu * dt / dy**2 * (vn[2:, 1:] - 2 * vn[1:, 1:] + vn[:-1, 1:]))
    # Boundary condition
    u_field[0, :] = 1
    u_field[-1, :] = 1
    u_field[:, 0] = 1
    u_field[:, -1] = 1
    v_field[0, :] = 1
    v_field[-1, :] = 1
    v_field[:, 0] = 1
    v_field[:, -1] = 1

# Visualisasi hasil
fig = plt.figure(figsize=(8, 6))
plt.contourf(x, y, u_field, cmap='jet')
plt.colorbar(label='Kecepatan u')
plt.title('Simulasi CFD 2D: Adveksi-Difusi')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
