
import numpy as np
import matplotlib.pyplot as plt

# Parameter simulasi 2D
Lx, Ly = 1.0, 1.0  # ukuran domain (meter)
Nx, Ny = 50, 50    # jumlah grid
Nt = 200           # jumlah langkah waktu
alpha = 1e-4       # difusivitas termal (m^2/s)
dx = Lx / (Nx - 1)
dy = Ly / (Ny - 1)
dt = 0.25 * min(dx, dy)**2 / alpha  # stabilitas eksplisit

# Inisialisasi grid
T = np.zeros((Nx, Ny))
# Sumber panas di tengah
T[Nx//2, Ny//2] = 100.0

# Simpan hasil untuk visualisasi
T_record = [T.copy()]

# Simulasi perambatan panas (2D eksplisit)
for n in range(Nt):
    T_new = T.copy()
    for i in range(1, Nx-1):
        for j in range(1, Ny-1):
            T_new[i, j] = T[i, j] + alpha * dt * (
                (T[i+1, j] - 2*T[i, j] + T[i-1, j]) / dx**2 +
                (T[i, j+1] - 2*T[i, j] + T[i, j-1]) / dy**2
            )
    # Boundary: suhu tetap 0
    T_new[0, :] = 0
    T_new[-1, :] = 0
    T_new[:, 0] = 0
    T_new[:, -1] = 0
    T = T_new
    if n % (Nt//5) == 0 or n == Nt-1:
        T_record.append(T.copy())

# Visualisasi hasil (snapshot)
fig, axes = plt.subplots(1, len(T_record), figsize=(3*len(T_record), 3))
if len(T_record) == 1:
    axes = [axes]
for ax, T_snap, step in zip(axes, T_record, range(len(T_record))):
    im = ax.imshow(T_snap.T, origin='lower', extent=[0, Lx, 0, Ly], cmap='hot')
    ax.set_title(f't={step*Nt//5*dt:.2f}s')
    plt.colorbar(im, ax=ax)
plt.suptitle('Simulasi Perambatan Panas 2D')
plt.tight_layout()
plt.savefig('thermal_simulation_2d.png')
plt.show()
