import qade
import numpy as np
from matplotlib import pyplot as plt

x = np.linspace(0, 1, 100)

u = qade.function(n_in=1, n_out=1)  # The function to be solved for

# wave equation evaluated at sample points (u[n,m,...] denotes nth derivative in first coordinate, mth in second etc)
diff_eq = qade.equation(u[2] + u[0], x)

# Boundary conditions for u at x = 0 and x = 1
a=1
b=0
x_left = 0
bc_left = qade.equation(u[1] - a, x_left)

x_right = x[-1]
bc_right = qade.equation(u[0] - b, x_right)

# Solve the equation using a basis of Fourier functions
solution = qade.solve(
    [diff_eq, bc_left, bc_right], qade.basis("monomial", 10), n_spins=20
)
print(f"loss = {solution.loss:.3}, weights = {np.around(solution.weights, 2)}")

def analytical_solution(x):  # analytical solution
    return a* np.sin(x) - a * np.cos(x)* np.sin(2) / (1 + np.cos(2))  + 2 *  b* np.cos(1) *np.cos(x) / (1 + np.cos(2))

plt.plot(x, solution(x), linewidth=1)
plt.plot(x, analytical_solution(x), linestyle="dashed", color="black")

plt.show()