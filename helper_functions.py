import numpy as np
from dimod import BinaryQuadraticModel, SPIN


def Pi_functional(S, a):
    """Returns the value of the Pi functional given S and a

    Args:
        S (N by 5 array): An array containing S_i for i from 1 to N
        a (array length N+1): An array containing the values at the nodes 0 to N

    Returns:
        float: value of the Pi functional
    """
    total = 0
    for S_i, A_i in zip(S, A_matrix(a)):
        total += np.dot(S_i, A_i)
    return total


# def k_i_to_index(k, i):
#     Not used anymore
#     """The qubits q_k^i need to be flattened into a vector for the bqm. This function computes the index of qubit
#     q_k^i given k and i
#     The qubits are ordered as follows: [q_1^0, q_1^1, ..., q_1^N, q_2^0, q_2^1, ..., q_2^N, q_3^0, q_3^1, ..., q_3^N]
#     (this makes it easier to unpack) i.e. q_k^i is found at index 3*(k-1) + i, with 1<=k<=3, 0<=i<=N

#     Args:
#         k (int): Number of qubit in node (1, 2 or 3)
#         i (int): Number of node (0 up to and including N)
#     """
#     # translate (k,i) pair to position in qubit vector
#     return 3 * (k - 1) + i


def mod_3(k):  # mod to range {1,2,3} instead of {0,1,2}
    return (k - 1) % 3 + 1


def get_label(k, i):  #
    """makes the label q_k^i given k, i

    Args:
        k (int): index of qubit (1, 2 or 3) at node i
        i (_type_): node index

    Returns:
        string: label q_k^i
    """
    return "q_{}^{}".format(k, i)


def parse_label(label):
    """Given a label of the form q_k^i, returns k, i as ints

    Args:
        label (string): label of the form q_k^i

    Returns:
        (int, int): k and i from the label
    """
    str_k, str_i = label[2:].split("^")  # ignore "q_", split at ^
    return int(str_k), int(str_i)


def create_bqm(H, J_hat, J_tildes, boundary_condition, b_c_strength=1):
    """Creates BinaryQuadraticModel from parameters

    Args:
        H (float): self-interaction strength
        J_hat (float): interaction strength between qubits in node graph
        J_tildes (array of matrices): interaction strengths between qubits of consecutive nodes
        boundary_condition (string): type of boundary condition: "D" for Dirichlet, "N" for Neumann, otherwise ignored
        b_c_strength (float): strength of boundary condition-enforcing interactions, the higher the stronger.
                              Coefficient will be set to the negative of this number

    Returns:
        BinaryQuadraticModel: bqm corresponding to the inputs
    """
    N = len(J_tildes)
    # are all J_tilde's the same? Then list is redundant, but N needs to be given

    # add self-interaction terms
    linear = {}
    for n in range(N + 1):
        for k in range(1, 4):
            linear[get_label(k, n)] = H

    quadratic = {}
    # add nodal graph interactions
    for i in range(N + 1):
        for k in range(1, 4):
            label1 = get_label(k, i)
            label2 = get_label(mod_3(k + 1), i)  # label of next qubit in 3-cycle
            quadratic[label1, label2] = J_hat

    # note that the nodal and element graphs share no edges
    # hence there is no risk of overwriting dict elements

    # add element graph interactions
    for i in range(1, N + 1):
        for k in range(1, 4):
            for l in range(1, 4):
                label1 = get_label(k, i - 1)
                label2 = get_label(l, i)
                quadratic[(label1, label2)] = J_tildes[i - 1][k - 1][l - 1]

    if boundary_condition in ["d", "D", "dirichlet", "Dirichlet"]:
        # Dirichlet boundary condition, see notes.txt
        linear[get_label(2, 0)] = -b_c_strength
        linear[get_label(2, N)] = -b_c_strength
    elif boundary_condition in ["n", "N", "neumann", "Neumann"]:
        # Neumann boundary condition, see notes.txt
        for k in range(1, 4):
            quadratic[(get_label(k, 0), get_label(k, 1))] = -b_c_strength
            quadratic[(get_label(k, N - 1), get_label(k, N))] = -b_c_strength

    return BinaryQuadraticModel(linear, quadratic, vartype=SPIN)


def qubit_triplet_to_value(q, v):
    """Given a triplet of qubits corresponding to one node and the set of allowed values, computes
    the associated value of the node, as in equation 11 of the paper

    Args:
        q ((±1, ±1, ±1)): Triplet of qubits at one node
        v (array of length 3): The allowed values of the node
    """
    total = 0
    for q_k, v_k in zip(q, v):
        total += (q_k + 1) / 2 * v_k
    # print(q)
    # try:
    #     idx = list(q).index(1)
    #     print("qtv:", idx + 1, v[idx], total, v)
    # except ValueError:
    #     print("Invalid qubit, got value", total)
    return total


def compute_a_min(sampleset, u_c, r):
    """Given a sampleset, returns the new best node values

    Args:
        sampleset (SampleSet): a sampleset coming from a dwave solver solution of a model created by create_bqm
        u_c (array of length N+1): the current best solution
        r (float): the slack variable

    Returns:
        array of length N+1: the node values corresponding to the best solution of the bqm
    """
    best_sample_dict = sampleset.first.sample
    N = len(best_sample_dict) // 3 - 1  # remember there are N+1 nodes
    qubit_array = np.zeros((N + 1, 3))
    for label, value in best_sample_dict.items():
        k, i = parse_label(label)
        qubit_array[i, k - 1] = value

    a_min = np.zeros(N + 1)
    for i in range(0, N + 1):
        a_min[i] = qubit_triplet_to_value(
            qubit_array[i], [u_c[i] - r, u_c[i], u_c[i] + r]
        )
    return a_min


def A_matrix(a_coefficients):
    """Computes the A_n vectors for each n and returns as a list of lists

    Args:
        a_coefficients (length N+1 array): the coefficients of the basis functions

    Returns:
        list of lists (N by 5): a list of A_n for each n
    """
    A = []
    for a_prev, a_current in zip(
        a_coefficients, a_coefficients[1:]
    ):  # get pairs (a_{i-1}, a_i)
        A.append([a_prev**2, a_current**2, a_prev * a_current, a_prev, a_current])
    return A


def compute_A(coeff_1, coeff_2):
    """Computes the A vector for two consecutive basis functions

    Args:
        coeff_1 (float): the coefficient of the first of the two basis functions
        coeff_2 (float): the coefficient of the second of the two basis functions

    Returns:
        _type_: _description_
    """
    return np.array([coeff_1**2, coeff_2**2, coeff_1 * coeff_2, coeff_1, coeff_2])


index_to_q_triplet = {
    1: (1, -1, -1),
    2: (-1, 1, -1),
    3: (-1, -1, 1),
}  # v_i_1 is encoded by q^i = (1,-1,-1) etc


def compute_J_tilde(n, S, v_values_prev, v_values_current):
    """Computes J tilde for the n-th element graph

    Args:
        n (int): index of element (1 up to & incl N)
        S (N by 5 array): the S vectors of the problem description
        v_values_prev (length 3 array): allowed values of node n-1
        v_values_current (length 3 array): allowed values of node n

    Returns:
        3x3 numpy array: J tilde of n-th element
    """
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
    """Computes the J tilde matrices for each of the element graphs.
    (Note that this differs from compute_J_tilde in that it computes the
    allowed values for each node given u_c and r, whereas the latter asks
    you to give the allowed values)

    Args:
        S (N by 5 array): the S vectors of the problem description
        u_c (length N+1 array): current best solution
        r (float): slack variable

    Returns:
        List of n 3x3 numpy arrays: the J tilde matrices
    """
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

def feasible_solution(sample):
    """"
    Takes a single sample and check whether it satisfies the condition  v^i_1 + v^i_2 + v^i_1 == -1 for all i

    :param sample:  The sample
    """
    # sample = sampleset.first.sample
    N = len(sample.sample) // 3 - 1  # remember there are N+1 nodes
    # print(sample.sample)
    qubit_array = np.zeros((N + 1, 3))
    for label, value in sample.sample.items():
        k, i = parse_label(label)
        qubit_array[i, k - 1] = value
    # print(  [sum(qubit_array[i]) for i in range(N+1)])
    for i in range(N+1):
        Sum = sum(qubit_array[i])
        if Sum != -1: # check whether v^i_1 + v^i_2 + v^i_1 == -1
            return False
    return True