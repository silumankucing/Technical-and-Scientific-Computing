import numpy as np
import matplotlib.pyplot as plt

# y(t) = 1 + ∫₀ᵗ (t-s) y(s) ds

t_max = 5
N = 200
t = np.linspace(0, t_max, N)
dt = t[1] - t[0]
y = np.zeros(N)

# Initial condition

y[0] = 1

# Numerical solution using the trapezoidal rule

for i in range(1, N):
    integrand = (t[i] - t[:i]) * y[:i]
    integral = np.trapz(integrand, t[:i])
    y[i] = 1 + integral

# Plot

plt.plot(t, y, label='Numerical Solution')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('Volterra Integral Equation Solution')
plt.legend()
plt.show()