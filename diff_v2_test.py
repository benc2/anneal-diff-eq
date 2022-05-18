import numpy as np
import matplotlib.pyplot as plt
from diff_eqn import SADiffEqn, LagrangeDiffEqn
import matplotlib.pyplot as plt

N = 6
r = 1.1 / N
r_min = 0.00001 * r
u_c = np.linspace(0, 1, N + 1) ** 2 * (1 - np.linspace(0, 1, N + 1))
diff_eq = SADiffEqn(
    p=1,
    q=0,
    f=2,
    initial_condition=u_c,
    nodes=N,
    basis_functions="triangle",
    boundary_condition="D",
)
# solution = diff_eq.solve(r=r, r_min=r_min)
# fig, ax = plt.subplots()
# diff_eq.plot_solution(ax=ax)


# def alpha(x):
#     arr = np.zeros((4, 3))
#     arr[3, 0] = 1
#     arr[0, 2] = -1
#     return arr

alpha = np.zeros((4, 3))
alpha[3, 0] = 1
alpha[0, 2] = -1

# alpha = np.arange(1, 10, 1).reshape((3, 3))
# x = np.expand_dims(alpha, (2, 3))
# y = np.ones((3,3,3,3))
# print(x*y)
# quit()

diff_eq = LagrangeDiffEqn(
    alpha=alpha, initial_condition=u_c, nodes=N, boundary_condition=None
)

solution = diff_eq.solve(r=r, r_min=r_min)
# soln_func = diff_eq.solution_function()
# x_axis = np.linspace(0, 1, 1000)
# plt.plot(x_axis, soln_func(x_axis))
diff_eq.plot_solution()
plt.show()

# input()
