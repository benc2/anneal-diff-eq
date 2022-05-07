import numpy as np
import matplotlib.pyplot as plt
import neal
from helper_functions import (
    Pi_functional,
    create_bqm,
    compute_a_min,
    compute_all_J_tildes,
)
from graph import show_bqm_graph


def simulated_sample(bqm):
    sim_solver = neal.SimulatedAnnealingSampler()
    return sim_solver.sample(bqm, beta_range=[0.1, 4.2])


if __name__ == "__main__":
    # N = None
    # r_min = None
    # S = None  # N by 5 array
    # r = None
    # u_c = None
    N = 2
    r_min = 0.1
    r = 0.5
    S = np.array([[1, 1, -2, 0, 0]] * N)
    u_c = np.array([0, 0.8, 1])

    H = 1
    J_hat = H  # set equal as in paper

    # here we need to do the embedding and stuff
    solver = None
    Pi_min = Pi_functional(S, u_c)

    # box algorithm
    while r > r_min:
        # print(r)
        J_tildes = compute_all_J_tildes(S, u_c, r)
        bqm = create_bqm(H, J_hat, J_tildes, boundary_condition="D")
        sampleset = simulated_sample(bqm)
        # solver.sample(bqm)  # adjust this to the solver!
        a_min = compute_a_min(sampleset, u_c, r)
        new_Pi = Pi_functional(S, a_min)
        if Pi_min < new_Pi:
            u_c = a_min
        else:
            r /= 2
            Pi_min = new_Pi

    plt.plot(np.linspace(0, 1, N + 1), u_c)
    plt.show()
