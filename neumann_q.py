import numpy as np
from scipy.special import i0, k0, i1, k1
from diff_eqn import NeumannSADiffEqn


def soln(x):
    return (
        (
            (np.sqrt(3) * k0(2 * np.sqrt(6)) - k1(2 * np.sqrt(3)))
            * i0(2 * np.sqrt(3 * (x + 1)))
            - (np.sqrt(3) * i0(2 * np.sqrt(6)) + i1(2 * np.sqrt(3)))
            * k0(2 * np.sqrt(3 * (x + 1)))
            + i0(2 * np.sqrt(6)) * k1(2 * np.sqrt(3))
            + i1(2 * np.sqrt(3)) * k0(2 * np.sqrt(6))
        )
        / 3
        / (
            i1(2 * np.sqrt(3)) * k0(2 * np.sqrt(6))
            + i0(2 * np.sqrt(6)) * k1(2 * np.sqrt(3))
        )
    )


N = 12
nodes = np.linspace(0, 1, N + 1)
# u_c = np.linspace(-0.3, 0, N + 1)
u_c = np.zeros(N + 1)
diff_eq = NeumannSADiffEqn(
    p=lambda x: (1 + x),
    q=3,
    f=1,
    neumann_side="l",
    neumann_value=1,
    initial_condition=u_c,
    nodes=nodes,
)
diff_eq.solve(r=1, r_min=1e-5)
diff_eq.animate(
    target_function=soln,
    y_bounds=(-0.4, 0.1),
)
