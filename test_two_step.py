"""This script is an example of how to implement the two-step improvement of the original algorithm using the
function solve_two_step from two_step.py. It solves the differential equation

    -u"(x) + x * u(x) = -2 * x

and plots it against the exact solution obtained from Wolfram Alpha.
"""

import numpy as np
from scipy.special import airy, gamma
import matplotlib.pyplot as plt

from two_step import solve_two_step

p = 1
q = lambda x: x
f = lambda x: -2 * x

# The exact solution (for reference)
solution = lambda x: (2 * np.pi * (airy(x)[0] * (
        gamma(1 / 3) * airy(x)[3] * (airy(1)[2] - np.sqrt(3) * airy(1)[0]) + np.sqrt(3) * gamma(1 / 3) * airy(1)[
    0] * airy(1)[3] - 3 ** (1 / 6) * airy(1)[2] * (2 + 3 ** (1 / 3) * gamma(1 / 3) * airy(1)[1])) + airy(x)[2] * (
                                           gamma(1 / 3) * airy(x)[1] * (np.sqrt(3) * airy(1)[0] - airy(1)[2]) +
                                           airy(1)[0] * (2 * 3 ** (1 / 6) - gamma(1 / 3) * airy(1)[3]) + gamma(
                                       1 / 3) * airy(1)[1] * airy(1)[2]))) / (
                             gamma(1 / 3) * (np.sqrt(3) * airy(1)[0] - airy(1)[2]))

N_first = 5
N_second = 20
u_init = np.zeros(N_first + 1)

step_one_diffeqn, step_two_diffeqn = solve_two_step(u_init, p=p, q=q, f=f, progress_bar=True,
                                                    return_objects=True, maxiter=200)

# Plot the results
x_axis = np.linspace(0, 1, 200)
plt.plot(x_axis, [solution(x) for x in x_axis], "k:")
step_one_diffeqn.plot_solution(color="red")
step_two_diffeqn.plot_solution(color="blue")
plt.legend(["Exact solution", "Step 1", "Step 2"])
plt.show()
