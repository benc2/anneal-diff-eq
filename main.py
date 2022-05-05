# the qubits q_k^i need to be flattened into a vector for the bqm
# they are ordered as follows: [q_1^0, q_1^1, ..., q_1^N, q_2^0, q_2^1, ..., q_2^N, q_3^0, q_3^1, ..., q_3^N]
# (this makes it easier to unpack) i.e. q_k^i is found at index 3*(k-1) + i, with 1<=k<=3, 0<=i<=N

from hashlib import new
import numpy as np
from dimod import BinaryQuadraticModel, SPIN

from compute_j_tilde import compute_all_J_tildes, A_matrix


def Pi_functional(S, a):
    total = 0
    for S_i, A_i in zip(S, A_matrix(a)):
        total += np.dot(S_i, A_i)
    return total


def k_i_to_index(k, i):  # translate (k,i) pair to position in qubit vector
    return 3 * i + k - 1


def mod_3(k):  # mod to range {1,2,3} instead of {0,1,2}
    return (k - 1) % 3 + 1


def create_bqm(H, J_hat, J_tildes):
    N = len(J_tildes)
    # are all J_tilde's the same? Then list is redundant, but N needs to be given
    linear = {n: H for n in range(N)}
    quadratic = {}
    for i in range(N):
        for k in range(1, 4):
            index = k_i_to_index(k, i)
            quadratic[(index, mod_3(index + 1))] = J_hat

    # note that the nodal and element graphs share no edges
    # hence there is no risk of overwriting dict elements
    for i in range(1, N):
        for k in range(1, 4):
            for l in range(1, 4):
                first_index = k_i_to_index(k, i - 1)
                second_index = k_i_to_index(l, i)
                quadratic[(first_index, second_index)] = J_tildes[i][k][l]

    return BinaryQuadraticModel(vartype=SPIN, linear=linear, quadratic=quadratic)


def qubit_triplet_to_value(q, v):  # see eqn 11
    total = 0
    for q_k, v_k in zip(q, v):
        total += (q_k + 1) / 2 * v_k
    return total


def compute_a_min(sampleset):
    best_solution_qubits = [value for key, value in sampleset.first.sample]
    # sampleset.first gives best solution, .sample gives dict {i: q_i}
    qubit_array = np.reshape(best_solution_qubits, (3, -1))
    # the rows are [q_k^0, q_k^1, ..., q_k^n]
    return [qubit_triplet_to_value(qubit_trip) for qubit_trip in qubit_array.T]


r_min = None
S = None  # N by 5 array
r = None
u_c = None

H = 1
J_hat = 1


# here we need to do the embedding and stuff
solver = None

Pi_min = Pi_functional(u_c)
while r > r_min:
    J_tildes = compute_all_J_tildes(S, u_c, r)
    bqm = create_bqm(H, J_hat, J_tildes)
    sampleset = solver.sample(bqm)  # adjust this to the solver!
    a_min = compute_a_min(sampleset)
    new_Pi = Pi_functional(a_min)
    if Pi_min < new_Pi:
        u_c = a_min
    else:
        r /= 2
        Pi_min = new_Pi
