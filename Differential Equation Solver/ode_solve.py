import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# dy/dt = -2y, with y(0) = 1

def dydt(t, y):
    return -2 * y

# Initial condition

y0 = [1]

# Time span for the solution

t_span = (0, 5)
t_eval = np.linspace(*t_span, 100)

# Solve

sol = solve_ivp(dydt, t_span, y0, t_eval=t_eval)

# Plot

plt.plot(sol.t, sol.y[0])
plt.xlabel('t')
plt.ylabel('y')
plt.title('Solution of dy/dt = -2y')
plt.show()