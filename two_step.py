# This script is an example of how to implement the two step improvement.

import numpy as np
from diff_eqn import SADiffEqn
from helper_functions import interpolate_nodes
from matplotlib.pyplot import show, plot, legend
from scipy.special import airy, gamma

# Set up the variables for the differential equation
# p, q and f determine the equation
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

# We set the (Dirichlet) boundary conditions
bc_left = 0
bc_right = 0

# We start with 6 nodes in the first step, and then upgrade to 21 nodes in the second step
N_first = 5
N_second = 20

# We define the values of r we want to use
r_first = 0.5
r_min_first = 0.01
r_second = r_min_first
r_min_second = 0.001 * r_second

# We initialize u_c for 6 nodes
u_init = np.linspace(bc_left, bc_right, N_first + 1)

# First we define an SADiffEqn object for the first step
step_one_diffeqn = SADiffEqn(
    p=p,
    q=q,
    f=f,
    initial_condition=u_init,
    nodes=N_first
    )

# We run the algoritm to solve the equation
first_solution = step_one_diffeqn.solve(r=r_first, r_min=r_min_first, progress_bar=True)
print("\nFirst step complete.")

# We interpolate the solution to get an initial guess for the second step
u_init_second = interpolate_nodes(first_solution, N_first, N_second)

# We define an SADiffEqn object for the second step
step_two_diffeqn = SADiffEqn(
    p=p,
    q=q,
    f=f,
    initial_condition=u_init_second,
    nodes=N_second
    )

# We run the algoritm to solve the equation
second_solution = step_two_diffeqn.solve(r=r_second, r_min=r_min_second, progress_bar=True)
print("\nSecond step complete.")

# Plot the two solutions
x_axis = np.linspace(0, 1, 200)
plot(x_axis, [solution(x) for x in x_axis], "k:")
step_one_diffeqn.plot_solution(color="red")
step_two_diffeqn.plot_solution(color="blue")
legend(["Exact solution", "Step 1", "Step 2"])
show()
