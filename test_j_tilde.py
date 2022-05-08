from helper_functions import (
    compute_J_tilde,
    A_matrix,
    qubit_triplet_to_value,
    index_to_q_triplet,
)
import numpy as np


def compute_LHS(qubit_triplets, discrete_value_sets, N):
    lhs_for_each_n = []
    for n in range(1, N + 1):
        J_tilde = compute_J_tilde(
            n, S, discrete_value_sets[n - 1], discrete_value_sets[n]
        )
        total = 0
        for k in range(3):
            for l in range(3):
                total += J_tilde[k, l] * qubit_triplets[n - 1][k] * qubit_triplets[n][l]
        lhs_for_each_n.append(total)
    return np.array(lhs_for_each_n)


def compute_RHS(qubit_triplets, discrete_value_sets, N):
    a_values = [
        qubit_triplet_to_value(q_trip, discrete_value_sets[i])
        for i, q_trip in enumerate(qubit_triplets)
    ]
    A = A_matrix(a_values)
    rhs_for_each_n = [np.dot(S[n - 1], A[n - 1]) for n in range(1, N + 1)]
    return np.array(rhs_for_each_n)


N = 5
random_ks = np.random.randint(low=1, high=4, size=N + 1)
qubit_triplets = [index_to_q_triplet[k] for k in random_ks]

discrete_value_sets = np.random.normal(scale=4, size=(N + 1, 3))  # the "v" values
S = np.random.normal(scale=7, size=(N, 5))


lhs = compute_LHS(qubit_triplets, discrete_value_sets, N)
rhs = compute_RHS(qubit_triplets, discrete_value_sets, N)
print(np.isclose(lhs, rhs))
