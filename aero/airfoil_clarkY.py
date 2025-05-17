import numpy as np
import matplotlib.pyplot as plt

def clark_y_camber_line(x):

  m = 0.2
  p = 0.4

  y_c = np.zeros_like(x)

  for i, xi in enumerate(x):
        if xi <= m:
            y_c[i] = p * (2*m*xi - xi**2) / m**2
        else:
            y_c[i] = p * (1 - 2*m + 2*m*xi - xi**2) / (1-m)**2

  return y_c


def clark_y_thickness_distribution(x):

    t_over_c = 0.12
    thickness = 4 * t_over_c * np.sqrt(x) * (1 - x)
    return thickness

def generate_clark_y_airfoil(x):

    y_c = clark_y_camber_line(x)
    thickness = clark_y_thickness_distribution(x)

    y_upper = y_c + thickness / 2
    y_lower = y_c - thickness / 2
    return y_upper, y_lower

x = np.linspace(0, 1, 100)

y_upper, y_lower = generate_clark_y_airfoil(x)

plt.figure(figsize=(8, 6))
plt.plot(x, y_upper, label="Upper Surface")
plt.plot(x, y_lower, label="Lower Surface")
plt.plot(x, clark_y_camber_line(x), '--', label="Camber Line")
plt.xlabel("x (Chord)")
plt.ylabel("y")
plt.title("Clark Y Airfoil")
plt.legend()
plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()