"""
File: eppler.py
Description: Generate Eppler airfoil coordinates (using digitized data or interpolation).
"""
import numpy as np

def eppler_airfoil_coordinates(name, n_points=100):
    """
    Generate coordinates for an Eppler airfoil using digitized data.
    Args:
        name (str): Eppler airfoil name, e.g. 'E193'
        n_points (int): Number of points along the chord
    Returns:
        tuple: (x, y_upper, y_lower)
    """
    # Example: Eppler E193 (replace with real data for other types)
    if name.upper() == 'E193':
        # Example digitized coordinates (x, y_upper, y_lower)
        # In practice, use full dataset or read from file
        x_raw = np.array([0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        y_upper_raw = np.array([0.0, 0.035, 0.05, 0.07, 0.08, 0.085, 0.087, 0.085, 0.08, 0.07, 0.045, 0.0])
        y_lower_raw = np.array([0.0, -0.015, -0.025, -0.035, -0.04, -0.042, -0.043, -0.042, -0.04, -0.035, -0.02, 0.0])
        x = np.linspace(0, 1, n_points)
        y_upper = np.interp(x, x_raw, y_upper_raw)
        y_lower = np.interp(x, x_raw, y_lower_raw)
        return x, y_upper, y_lower
    else:
        raise ValueError(f"Eppler airfoil '{name}' not available in this example.")

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    name = "E193"
    x, y_upper, y_lower = eppler_airfoil_coordinates(name)
    plt.plot(x, y_upper, 'b-', label='Upper Surface')
    plt.plot(x, y_lower, 'r-', label='Lower Surface')
    plt.title(f'Eppler {name} Airfoil')
    plt.axis('equal')
    plt.legend()
    plt.show()
