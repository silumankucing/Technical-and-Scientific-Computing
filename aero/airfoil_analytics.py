import numpy as np
import matplotlib.pyplot as plt

def display_parameters(airfoil_type, code, m=None, p=None, t=None, cl=None, series=None, a=None):
    print(f"1. Chamber: {m * 100 if m is not None else 'N/A'}% of chord")
    print(f"2. Thickness: {t * 100 if t is not None else 'N/A'}% of chord")
    print(f"3. Chord Length: 1 (normalized)")
    print(f"4. Angle of Attack: User-defined (not calculated here)")
    print(f"5. Leading Edge Radius: Estimated based on thickness")
    print(f"6. Trailing Edge Angle: Estimated based on thickness")
    print(f"7. Maximum Thickness Location: {p * 100 if p is not None else 'N/A'}% of chord")
    print(f"8. Lift Coefficient (CL): {cl if cl is not None else 'N/A'}")
    print(f"9. Drag Coefficient (CD): Estimated based on thickness and camber")
    print(f"10. Lift-to-Drag Ratio (L/D): Estimated based on CL and CD")
    print(f"11. Reynolds Number (Re): User-defined (not calculated here)")
    print(f"12. Stall Angle: Estimated based on camber and thickness")
    print(f"13. Pressure Distribution: Not calculated here")
    print(f"14. Moment Coefficient (Cm): Estimated based on camber")
    print(f"15. Transition Point: Not calculated here")

def main():
    print("Available types: 4-digit, 5-digit, 6-series")
    code = input("Enter airfoil code (e.g., 2412 for 4-digit): ").strip()

    display_parameters(airfoil_type, code, m, p, t, cl, series, a)

if __name__ == "__main__":
    main()