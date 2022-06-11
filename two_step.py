# This script is an example of how to implement the two step improvement. Alternatively, one could import the
# function solve_two_step for a straightforward implementation

import numpy as np
from diff_eqn import SADiffEqn
from helper_functions import interpolate_nodes
import matplotlib.pyplot as plt
from scipy.special import airy, gamma
from warnings import warn


if __name__ == "__main__":
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
    print("Executing first step:")
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
    print("Executing second step:")
    second_solution = step_two_diffeqn.solve(r=r_second, r_min=r_min_second, progress_bar=True)
    print("\nSecond step complete.")

    # Plot the two solutions
    x_axis = np.linspace(0, 1, 200)
    plt.plot(x_axis, [solution(x) for x in x_axis], "k:")
    step_one_diffeqn.plot_solution(color="red")
    step_two_diffeqn.plot_solution(color="blue")
    plt.legend(["Exact solution", "Step 1", "Step 2"])
    plt.show()
else:
    def solve_two_step(
            initial_condition,
            p=1,
            q=0,
            f=0,
            nodes_first=5,
            nodes_second=20,
            progress_bar=True,
            r_first=0.5,
            r_min_first=None,
            r_second=0.01,
            r_min_second=None,
            maxiter=np.inf,
            return_objects=False
    ):
        """

        :param initial_condition:
        :param p: Int or function p
        :param q: Int or function q
        :param f: Int or function f
        :param nodes_first:
        :param nodes_second:
        :param progress_bar:
        :param r_first:
        :param r_min_first:
        :param r_second:
        :param r_min_second:
        :param maxiter:
        :param return_objects: If True, the function returns the SADiffEqn objects, and returns the solution otherwise.
        :return: tuple of SADiffEqn objects, or solution
        """
        # We check some values
        if not isinstance(r_first, float) or r_first <= 0:
            warn("'r_first' should be a positive scalar, resetting to default value.")
            r_first = 0.5
        if not isinstance(r_min_first, float) or r_first <= r_min_first or r_min_first <= 0:
            if r_min_first is None:
                pass
            else:
                warn("'r_first_min' has invalid value. Require 0<r_first_min<r_first, resetting to default value.")
            r_min_first = 0.01 * r_first

        if not isinstance(r_second, float) or r_second <= 0:
            warn("'r_second' should be a positive scalar, resetting to default value.")
            r_first = 0.5
        if not isinstance(r_min_second, float) or r_second <= r_min_second or r_min_second <= 0:
            if r_min_second is None:
                pass
            else:
                warn("'r_first_min' has invalid value. Require 0<r_first_min<r_first, resetting to default value.")
            r_min_second = 0.0005 * r_second

        # We define an SADiffEqn object for the first step
        step_one_diffeqn = SADiffEqn(
            p=p,
            q=q,
            f=f,
            initial_condition=initial_condition,
            nodes=nodes_first
        )

        # We run the algoritm to solve the equation
        if progress_bar:
            print("Executing first step:")
        first_solution = step_one_diffeqn.solve(r=r_first, r_min=r_min_first, progress_bar=progress_bar,
                                                maxiter=maxiter)
        if progress_bar:
            print("\nFirst step complete.")

        # We interpolate the solution to get an initial guess for the second step
        u_init_second = interpolate_nodes(first_solution, nodes_first, nodes_second)

        # We update maxiter:
        maxiter = maxiter - step_one_diffeqn.i
        # We define an SADiffEqn object for the second step
        step_two_diffeqn = SADiffEqn(
            p=p,
            q=q,
            f=f,
            initial_condition=u_init_second,
            nodes=nodes_second
        )

        # We run the algoritm to solve the equation
        if progress_bar:
            print("Executing second step:")
        second_solution = step_two_diffeqn.solve(r=r_second, r_min=r_min_second, progress_bar=progress_bar,
                                                 maxiter=maxiter)
        if progress_bar:
            print("\nSecond step complete.")

        if not return_objects:
            return second_solution
        else:
            return step_one_diffeqn, step_two_diffeqn
