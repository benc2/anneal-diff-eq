import numpy as np

# S = [[1, 1, -2, 0, 0]] * 2
# v = [0, 0.5, 1]


def A_matrix(node_values):
    A = []
    for a_prev, a_current in zip(
        node_values, node_values[1:]
    ):  # get pairs (a_{i-1}, a_i)
        A.append([a_prev**2, a_current**2, a_prev * a_current, a_prev, a_current])
    return A


def compute_A(node_1, node_2):
    return np.array([node_1**2, node_2**2, node_1 * node_2, node_1, node_2])


index_to_q_triplet = {
    1: (1, -1, -1),
    2: (-1, 1, -1),
    3: (-1, -1, 1),
}  # v_i_1 is encoded by q^i = (1,-1,-1) etc


def compute_J_tilde(n, S, v_values_prev, v_values_current):
    # n starting from 1
    matrix = []
    b = []
    for i in range(1, 4):
        for j in range(1, 4):
            row = []
            q_i = index_to_q_triplet[i]
            q_j = index_to_q_triplet[j]
            for k in range(3):
                for l in range(3):
                    row.append(q_i[k] * q_j[l])
            matrix.append(row)

            b.append(
                np.dot(
                    compute_A(v_values_prev[i - 1], v_values_current[j - 1]), S[n - 1]
                )
            )

    matrix = np.array(matrix).T
    J_vector = np.linalg.solve(matrix, b)  # do we need to transpose the matrix?
    # J_matrix = []
    # for i in range(3):
    #     J_matrix.append([J_vector[3*i + j] for j in range(3)])

    J_matrix = J_vector.reshape((3, 3))

    return np.array(J_matrix)


def compute_all_J_tildes(S, u_c, r):
    N = len(u_c) - 1
    j_tildes = []
    for i in range(1, N + 1):
        prev_node = u_c[i - 1]
        curr_node = u_c[i]
        new_j_tilde = compute_J_tilde(
            i,
            S,
            [prev_node - r, prev_node, prev_node + r],
            [curr_node - r, curr_node, curr_node + r],
        )
        j_tildes.append(new_j_tilde)
    return j_tildes


# print(compute_J_tilde(1, v, v))
