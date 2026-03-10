import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parameter geometris bracket sederhana (2D)
length = 0.2  # meter
height = 0.05  # meter
thickness = 0.01  # meter
E = 210e9  # Young's modulus (Pa) untuk baja
nu = 0.3  # Poisson's ratio
F = 1000  # Gaya (N)

# Mesh sederhana (4 node, 2 elemen)
# Node: (x, y)
nodes = np.array([
    [0, 0],
    [length, 0],
    [0, height],
    [length, height]
])

# Elemen: (node1, node2, node3, node4)
elements = [
    [0, 1, 3, 2]
]

# Matriks kekakuan elemen (plane stress, 4 node quad, simplified)
def element_stiffness(E, nu, t, l, h):
    # Matriks kekakuan sederhana untuk demo (bukan FEM penuh)
    k = E * t * h / l
    return k * np.array([
        [1, -1],
        [-1, 1]
    ])

# Hitung deformasi (asumsi: node 0 dan 2 fixed, gaya di node 1 dan 3)
K = np.zeros((4, 4))
ke = element_stiffness(E, nu, thickness, length, height)
K[1:3, 1:3] += ke

# Gaya eksternal
F_ext = np.zeros(4)
F_ext[1] = F / 2
F_ext[3] = F / 2

# Boundary condition: node 0 dan 2 fixed
free_dof = [1, 3]
K_reduced = K[np.ix_(free_dof, free_dof)]
F_reduced = F_ext[free_dof]


# Regularisasi agar K_reduced tidak singular (khusus demo visualisasi)
reg = 1e-8 * np.eye(K_reduced.shape[0])
u = np.linalg.solve(K_reduced + reg, F_reduced)

# Update posisi node
nodes_deformed = nodes.copy()
nodes_deformed[1, 0] += u[0] * 1000  # skala visualisasi
nodes_deformed[3, 0] += u[1] * 1000

# Hitung stress (sigma = F/A)
A = thickness * height
stress = F / A
strain = stress / E


print(f"Stress: {stress:.2f} Pa")
print(f"Strain: {strain:.6e}")
print(f"Deformasi node 1: {u[0]:.6e} m")
print(f"Deformasi node 3: {u[1]:.6e} m")

# --- VTK Output ---
def write_vtk(nodes, elements, u, filename="bracket2d.vtk"):
    # VTK legacy format (ASCII, POLYDATA)
    with open(filename, 'w') as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Bracket2D Simulation\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")
        f.write(f"POINTS {len(nodes)} float\n")
        for i, node in enumerate(nodes):
            # Z=0 for 2D
            f.write(f"{node[0]:.6e} {node[1]:.6e} 0.0\n")
        f.write(f"POLYGONS {len(elements)} {len(elements)*(len(elements[0])+1)}\n")
        for elem in elements:
            f.write(f"{len(elem)} {' '.join(str(i) for i in elem)}\n")
        # Nodal displacement as point data
        f.write(f"\nPOINT_DATA {len(nodes)}\n")
        f.write("SCALARS displacement float 1\nLOOKUP_TABLE default\n")
        # Only x-displacement for free nodes, 0 for fixed
        disp = [0, u[0], 0, u[1]]
        for d in disp:
            f.write(f"{d:.6e}\n")
        f.write("\n")
        f.write("SCALARS stress float 1\nLOOKUP_TABLE default\n")
        for _ in nodes:
            f.write(f"{stress:.6e}\n")
        f.write("\n")
    print(f"VTK file written: {filename}")

write_vtk(nodes, elements, u)

# Visualisasi statis
def plot_bracket(nodes, nodes_def, title):
    fig, ax = plt.subplots()
    for elem in elements:
        orig = nodes[elem + [elem[0]], :]
        defo = nodes_def[elem + [elem[0]], :]
        ax.plot(orig[:, 0], orig[:, 1], 'b--', label='Asli' if elem == elements[0] else "")
        ax.plot(defo[:, 0], defo[:, 1], 'r-', label='Terdeformasi' if elem == elements[0] else "")
    ax.legend()
    ax.set_aspect('equal')
    ax.set_title(title)
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    try:
        plt.show()
    except Exception as e:
        outname = "bracket2d_plot.png"
        plt.savefig(outname)
        print(f"Plot statis tidak interaktif, hasil disimpan ke: {outname}")


plot_bracket(nodes, nodes_deformed, "Simulasi Bracket 2D: Asli vs Terdeformasi")

# --- VTK Viewer Sederhana (matplotlib) ---
def view_vtk(filename="bracket2d.vtk"):
    # Hanya untuk legacy VTK POLYDATA sederhana
    pts = []
    polys = []
    disp = []
    with open(filename) as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        if lines[i].startswith("POINTS"):
            n = int(lines[i].split()[1])
            for j in range(n):
                pts.append([float(x) for x in lines[i+1+j].split()[:2]])
            i += n+1
        elif lines[i].startswith("POLYGONS"):
            n = int(lines[i].split()[1])
            for j in range(n):
                idx = [int(x) for x in lines[i+1+j].split()[1:]]
                polys.append(idx)
            i += n+1
        elif lines[i].startswith("SCALARS displacement"):
            i += 2
            for j in range(len(pts)):
                disp.append(float(lines[i+j]))
            i += len(pts)
        else:
            i += 1
    pts = np.array(pts)
    disp = np.array(disp)
    fig, ax = plt.subplots()
    for poly in polys:
        poly_pts = pts[poly]
        c = ax.fill(poly_pts[:,0], poly_pts[:,1], facecolor='none', edgecolor='k', lw=2)
        # Color by displacement (optional)
        for (x, y, d) in zip(poly_pts[:,0], poly_pts[:,1], disp[poly]):
            ax.scatter(x, y, c='r' if d!=0 else 'b', s=60)
    sc = ax.scatter(pts[:,0], pts[:,1], c=disp, cmap='jet', s=100, zorder=3)
    plt.colorbar(sc, label='Displacement (m)')
    ax.set_aspect('equal')
    ax.set_title('VTK Viewer: Displacement')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.show()


# Jalankan viewer VTK secara otomatis setelah simulasi
view_vtk("bracket2d.vtk")

# Animasi deformasi
def animate_deformation(nodes, nodes_def):
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_xlim(-0.01, length + 0.03)
    ax.set_ylim(-0.01, height + 0.01)
    line_orig, = ax.plot([], [], 'b--', label='Asli')
    line_def, = ax.plot([], [], 'r-', label='Terdeformasi')
    ax.legend()
    
    def init():
        line_orig.set_data([], [])
        line_def.set_data([], [])
        return line_orig, line_def
    
    def update(frame):
        alpha = frame / 20
        interp = nodes + alpha * (nodes_def - nodes)
        for elem in elements:
            orig = nodes[elem + [elem[0]], :]
            defo = interp[elem + [elem[0]], :]
            line_orig.set_data(orig[:, 0], orig[:, 1])
            line_def.set_data(defo[:, 0], defo[:, 1])
        return line_orig, line_def
    
    ani = animation.FuncAnimation(fig, update, frames=21, init_func=init, blit=True, interval=100)
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('Animasi Deformasi Bracket 2D')
    try:
        plt.show()
    except Exception as e:
        outname = "bracket2d_animasi.png"
        ani.save(outname)
        print(f"Animasi tidak interaktif, hasil disimpan ke: {outname}")

animate_deformation(nodes, nodes_deformed)
