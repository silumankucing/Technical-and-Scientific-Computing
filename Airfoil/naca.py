import numpy as np
import matplotlib.pyplot as plt

def naca4_airfoil_points(code, n_points=100):
    # code: string, e.g. '2412'
    m = int(code[0]) / 100.0
    p = int(code[1]) / 10.0
    t = int(code[2:]) / 100.0
    # x-coordinates (cosine spacing)
    x = (1 - np.cos(np.linspace(0, np.pi, n_points))) / 2
    # Thickness distribution
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
    # Camber line
    yc = np.zeros_like(x)
    dyc_dx = np.zeros_like(x)
    for i, xi in enumerate(x):
        if xi < p and p != 0:
            yc[i] = m / p**2 * (2*p*xi - xi**2)
            dyc_dx[i] = 2*m/p**2 * (p - xi)
        elif p != 0:
            yc[i] = m / (1-p)**2 * ((1-2*p) + 2*p*xi - xi**2)
            dyc_dx[i] = 2*m/(1-p)**2 * (p - xi)
    theta = np.arctan(dyc_dx)
    # Upper and lower surfaces
    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)
    # Combine upper and lower
    x_points = np.concatenate([xu[::-1], xl[1:]])
    y_points = np.concatenate([yu[::-1], yl[1:]])
    return x_points, y_points

def main():
    code = input("Masukkan kode NACA 4-digit (misal 2412): ").strip()
    n_points = input("Jumlah titik (default 100): ").strip()
    if not n_points:
        n_points = 100
    else:
        n_points = int(n_points)
    x, y = naca4_airfoil_points(code, n_points)
    plt.figure(figsize=(8,2))
    plt.plot(x, y, '-k')
    plt.axis('equal')
    plt.title(f'NACA {code} Airfoil')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'naca_{code}.png')
    plt.show()

if __name__ == "__main__":
    main()
