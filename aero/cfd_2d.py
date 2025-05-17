import numpy as np
import matplotlib.pyplot as plt

# Constants
N_PANELS = 100  # Number of panels
CHORD_LENGTH = 1.0  # Chord length
ANGLE_OF_ATTACK = 5.0  # Angle of attack in degrees
V_INFINITY = 1.0  # Freestream velocity

# Generate NACA 4-digit airfoil points
def generate_naca_4digit_airfoil(m, p, t, n_panels, chord_length):
    x = np.linspace(0, chord_length, n_panels + 1)
    yt = 5 * t * (0.2969 * np.sqrt(x / chord_length) - 0.1260 * (x / chord_length) -
                  0.3516 * (x / chord_length)**2 + 0.2843 * (x / chord_length)**3 -
                  0.1015 * (x / chord_length)**4)
    return x, yt

# Vortex panel method solver
def vortex_panel_method(x, y, angle_of_attack, v_infinity):
    n_panels = len(x) - 1
    alpha = np.radians(angle_of_attack)  # Angle of attack in radians

    # Panel coordinates
    x_mid = (x[:-1] + x[1:]) / 2  # Midpoints of panels
    y_mid = (y[:-1] + y[1:]) / 2
    lengths = np.sqrt((x[1:] - x[:-1])**2 + (y[1:] - y[:-1])**2)  # Panel lengths
    theta = np.arctan2(y[1:] - y[:-1], x[1:] - x[:-1])  # Panel angles

    # Initialize influence coefficient matrix and RHS vector
    A = np.zeros((n_panels, n_panels))
    b = np.zeros(n_panels)

    # Calculate influence coefficients
    for i in range(n_panels):
        for j in range(n_panels):
            if i == j:
                A[i, j] = 0.5  # Self-influence
            else:
                # Influence of panel j on panel i
                x_diff = x_mid[i] - x[j]
                y_diff = y_mid[i] - y[j]
                r_squared = x_diff**2 + y_diff**2
                A[i, j] = (1 / (2 * np.pi)) * (y_diff / r_squared)

        # RHS vector (normal component of freestream velocity)
        b[i] = -v_infinity * np.sin(theta[i] - alpha)

    # Solve for vortex strengths
    gamma = np.linalg.solve(A, b)

    # Calculate pressure coefficients
    cp = 1 - (gamma / v_infinity)**2

    return x_mid, cp, gamma

# Plot pressure distribution
def plot_pressure_distribution(x, cp):
    plt.figure(figsize=(10, 6))
    plt.plot(x, cp, '-o', label="Pressure Coefficient (Cp)")
    plt.title("Pressure Distribution on Airfoil")
    plt.xlabel("Chord Position (x)")
    plt.ylabel("Pressure Coefficient (Cp)")
    plt.gca().invert_yaxis()  # Invert y-axis for better visualization
    plt.grid(True)
    plt.legend()
    plt.show()

# Plot streamlines
def plot_streamlines(x, y, gamma, v_infinity):
    # Create a grid for streamlines
    x_grid, y_grid = np.meshgrid(np.linspace(-0.5, 1.5, 100), np.linspace(-0.5, 0.5, 100))
    u = v_infinity * np.ones_like(x_grid)  # x-component of velocity
    v = np.zeros_like(y_grid)  # y-component of velocity

    # Add influence of vortex panels
    for i in range(len(x) - 1):
        x_panel = (x[i] + x[i + 1]) / 2
        y_panel = (y[i] + y[i + 1]) / 2
        r_squared = (x_grid - x_panel)**2 + (y_grid - y_panel)**2
        u += (gamma[i] / (2 * np.pi)) * (y_grid - y_panel) / r_squared
        v -= (gamma[i] / (2 * np.pi)) * (x_grid - x_panel) / r_squared

    # Plot streamlines
    plt.figure(figsize=(10, 6))
    plt.streamplot(x_grid, y_grid, u, v, density=2, color='b', linewidth=1)
    plt.fill(x, y, 'k', label="Airfoil")  # Plot airfoil
    plt.title("Streamlines Around Airfoil")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.show()

# Main function
def main():
    # Generate airfoil geometry (NACA 0012)
    x, y = generate_naca_4digit_airfoil(0, 0, 12, N_PANELS, CHORD_LENGTH)

    # Solve for pressure distribution and vortex strengths
    x_mid, cp, gamma = vortex_panel_method(x, y, ANGLE_OF_ATTACK, V_INFINITY)

    # Visualize results
    plot_pressure_distribution(x_mid, cp)
    plot_streamlines(x, y, gamma, V_INFINITY)

if __name__ == "__main__":
    main()