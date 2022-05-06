import numpy as np
import matplotlib.pyplot as plt
import neal
from dimod import BinaryQuadraticModel, SPIN
from compute_j_tilde import compute_all_J_tildes, A_matrix


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
    for i in range(N):
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
        for k in range(3):
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
    qubit_array = np.zeros((3, N + 1))
    for label, value in best_sample_dict.items():
        k, i = parse_label(label)
        qubit_array[k - 1, i] = value

    a_min = np.zeros(N + 1)
    for i in range(0, N + 1):
        a_min[i] = qubit_triplet_to_value(
            qubit_array[i], [u_c[i] - r, u_c[i], u_c[i] + r]
        )
    return a_min


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
    J_hat = 1

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
