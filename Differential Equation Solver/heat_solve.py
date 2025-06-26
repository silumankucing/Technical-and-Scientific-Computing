import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

k = 0.3                # Cooling constant
T_env = 25             # Ambient temperature (°C)
T0 = 90                # Initial temperature (°C)

# dT/dt = -k * (T - T_env)

def dTdt(t, T):
    return -k * (T - T_env)

t_span = (0, 20)
t_eval = np.linspace(*t_span, 200)

# Solve

sol = solve_ivp(dTdt, t_span, [T0], t_eval=t_eval)

# Plot

plt.plot(sol.t, sol.y[0], label='Object Temperature')
plt.axhline(T_env, color='r', linestyle='--', label='Ambient Temperature')
plt.xlabel('Time (minutes)')
plt.ylabel('Temperature (°C)')
plt.title("Newton's Law of Cooling")
plt.legend()
plt.show()