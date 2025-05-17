import numpy as np
import matplotlib.pyplot as plt

def naca4_digit(code):

    m = int(code[0]) / 100
    p = int(code[1]) / 10
    t = int(code[2:]) / 100

    x = np.linspace(0, 1, 200)
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
    yc = np.where(x < p, m / p**2 * (2 * p * x - x**2), m / (1 - p)**2 * ((1 - 2 * p) + 2 * p * x - x**2))
    
    theta = np.arctan(np.gradient(yc, x))
    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)

    return xu, yu, xl, yl

def naca5_digit(code):

    cl = int(code[0]) * 0.15
    p = int(code[1]) / 20
    t = int(code[2:]) / 100

    x = np.linspace(0, 1, 200)
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
    yc = np.where(x < p, cl / 6 * (x**3 - 3 * p * x**2 + p**2 * (3 - p) * x), cl * p**3 / 6 * (1 - x))
    
    theta = np.arctan(np.gradient(yc, x))
    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)

    return xu, yu, xl, yl

def naca6_series(code):

    series = int(code[0])
    a = int(code[1]) / 10
    cl = int(code[2]) * 0.1
    t = int(code[3:]) / 100

    x = np.linspace(0, 1, 200)
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)

    yc = cl / (2 * np.pi * (a + 1)) * (np.log(x) - x)
    theta = np.arctan(np.gradient(yc, x))
    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)

    return xu, yu, xl, yl

def plot_naca_airfoil(xu, yu, xl, yl, title):

    plt.figure(figsize=(10, 4))
    plt.plot(xu, yu, label='Upper Surface')
    plt.plot(xl, yl, label='Lower Surface')
    plt.title(title)
    plt.xlabel('Chord Position (x/c)')
    plt.ylabel('Thickness (y/c)')
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.show()

def main():
    print("NACA Airfoil Generator")
    print("Available types: 4-digit, 5-digit, 6-series")
    airfoil_type = input("Enter airfoil type (4/5/6): ").strip()
    code = input("Enter airfoil code (e.g., 2412 for 4-digit): ").strip()

    if airfoil_type == '4':
        xu, yu, xl, yl = naca4_digit(code)
        title = f"NACA {code} (4-Digit Airfoil)"
    elif airfoil_type == '5':
        xu, yu, xl, yl = naca5_digit(code)
        title = f"NACA {code} (5-Digit Airfoil)"
    elif airfoil_type == '6':
        xu, yu, xl, yl = naca6_series(code)
        title = f"NACA {code} (6-Series Airfoil)"
    else:
        print("Invalid airfoil type. Please choose 4, 5, or 6.")
        return

    plot_naca_airfoil(xu, yu, xl, yl, title)

if __name__ == "__main__":
    main()