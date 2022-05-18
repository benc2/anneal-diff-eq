import numpy as np
import matplotlib.pyplot as plt
import neal
from dwave.system import DWaveSampler, EmbeddingComposite, LeapHybridSampler

from helper_functions import (
    Pi_functional,
    create_bqm,
    compute_a_min,
    compute_all_J_tildes,
    feasible_solution,
)
from graph import show_bqm_graph
from basisfunctions import calculate_S


def simulated_sample(bqm, filter=False):
    ready = False
    while not ready:
        sim_solver = neal.SimulatedAnnealingSampler()
        sampleset = sim_solver.sample(bqm, num_reads=1000).aggregate()
        if filter:
            sampleset = sampleset.filter(feasible_solution)
            if len(sampleset) > 0:
                ready = True
        else:
            ready = True
    return sampleset


def real_sample(bqm, filter=False):
    real_solver = EmbeddingComposite(DWaveSampler())
    return real_solver.sample(
        bqm, num_reads=1000, annealing_time=100, return_embedding=True
    ).aggregate()

def hybrid_sample(bqm, filter=False):
    real_solver = LeapHybridSampler()
    return real_solver.sample(
        bqm, time_limit=5
    ).aggregate()


if __name__ == "__main__":
    # N = None
    # r_min = None
    # S = None  # N by 5 array
    # r = None
    # u_c = None
    N = 100
    r_min = 0.002
    r = 1
    # S = np.array([[1, 1, -2, 0, 0]] * N)
    u_c = np.array(np.linspace(0, 1, N + 1)) ** 0.7
    S = calculate_S(N, p=1, q=0, f=0)  # S depends on the distance between the nodes

    H = 1
    J_hat = H  # set equal as in paper

    # here we need to do the embedding and stuff
    solver = None
    # Pi_min = Pi_functional(S, u_c)

    # box algorithm
    while r > r_min:
        J_tildes = compute_all_J_tildes(S, u_c, r)
        bqm = create_bqm(H, J_hat, J_tildes, boundary_condition="D", b_c_strength=1)
        sampleset = simulated_sample(bqm, filter=False)

        # solver.sample(bqm)  # adjust this to the solver!
        a_min = compute_a_min(sampleset, u_c, r)
        # new_Pi = Pi_functional(S, a_min)
        if Pi_functional(S, a_min) < Pi_functional(S, u_c):
            u_c = a_min
        else:
            r /= 2
            # Pi_min = new_Pi

    plt.plot(np.linspace(0, 1, N + 1), u_c)
    plt.show()
