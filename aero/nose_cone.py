import numpy as np
import matplotlib.pyplot as plt

def generate_nose_cone(cone_type, length, radius, num_points=100):
    x = np.linspace(0, length, num_points)
    
    if cone_type == "conical":
        y = (radius / length) * x
    elif cone_type == "ogive":
        rho = (radius**2 + length**2) / (2 * radius)
        y = np.sqrt(rho**2 - (length - x)**2) - (rho - radius)
    elif cone_type == "parabolic":
        y = radius * (1 - (x / length)**2)
    else:
        raise ValueError("Invalid cone type")
    
    return x, y

cone_type = input("Enter the type of nose cone (conical, ogive, parabolic): ").lower()

length = float(input("Enter the length of the nose cone: "))
radius = float(input("Enter the radius of the nose cone: "))

try:
    x, y = generate_nose_cone(cone_type, length, radius)

    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label=f"{cone_type.capitalize()} Nose Cone")
    plt.plot(x, -y, linestyle="--", color="blue")
    plt.title(f"{cone_type.capitalize()} Nose Cone Shape")
    plt.xlabel("Length")
    plt.ylabel("Radius")
    plt.grid(True)
    plt.axis("equal")
    plt.legend()
    plt.show()

except ValueError as e:
    print(e)