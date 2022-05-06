import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.misc import derivative


def tent(x, start, peak, end):
    if x <= start or x > end:
        return 0
    elif start < x <= peak:
        return (x - start) / (peak - start)
    elif peak < x <= end:
        return 1 - (x - peak) / (end - peak)


def basis_functions(nodes, shape="triangle", x_l=0, x_r=1):
    # give int to get equally spaced points, or give nodes directly
    if isinstance(nodes, int):
        nodes = np.linspace(x_l, x_r, nodes + 1)

    # def phi_evaluation(i, x):   Dit klopt niet volgens mij, de eerste basis-functie heeft phi(0)=1 anders kan je nooit een waarde anders dan nul hebben in het punt 0
    #     return tent(x, nodes[i], nodes[i + 1], nodes[i + 2]) #de nodes moeten eentje verschoven, zie hieronder
    def phi_evaluation(i, x):
        if i == 0:
            return tent(x, nodes[i] - 1, nodes[i], nodes[i + 1])
        if i == len(nodes) - 1:
            return tent(x, nodes[i - 1], nodes[i], nodes[i] + 1)
        else:
            return tent(x, nodes[i - 1], nodes[i], nodes[i + 1])

    phi_output = np.vectorize(
        phi_evaluation, otypes=[float]
    )  # makes it so you can input an array

    return phi_output


def calculate_S(nodes, basis, x_l=0, x_r=1, p=1, q=0, f=0):
    """
    Calculates S for a given set of nodes and input functions
    :type nodes: int or array
    :param nodes: Either an integer (for evenly spaced nodes) or an array of custom nodes
    :param basis: Basis functions
    :type x_l: float
    :type x_r: float
    :param x_l: Left boundary of the range (if nodes is an integer)
    :param x_r: Right boundary of the range (if nodes is an integer)
    :type p: float or function
    :type q: float or function
    :type f: float or function
    """
    if isinstance(nodes, int):
        nodes = np.linspace(x_l, x_r, nodes + 1)
    if isinstance(p, (int, float)):
        p_val = p
        p = lambda x: p_val
    if isinstance(q, (int, float)):
        q_val = q
        q = lambda x: q_val
    if isinstance(f, (int, float)):
        f_val = f
        f = lambda x: f_val
    s = np.ndarray((len(nodes) - 1, 5))
    for i in range(len(nodes) - 1):  # i=0  here represents i=1 in the article
        i = i + 1
        phi_i = lambda x: basis(i, x)
        phi_i_minus_1 = lambda x: basis(i - 1, x)
        phi_i_prime = lambda x: derivative(phi_i, x, 0.00001)
        phi_i_minus_1_prime = lambda x: derivative(phi_i_minus_1, x, 0.00001)
        s[i - 1, :] = [
            integrate.quad(
                lambda x: 1 / 2 * p(x) * phi_i_minus_1_prime(x) ** 2
                + 1 / 2 * q(x) * phi_i_minus_1(x) ** 2,
                nodes[i - 1],
                nodes[i],
            )[0],
            integrate.quad(
                lambda x: 1 / 2 * p(x) * phi_i_prime(x) ** 2
                + 1 / 2 * q(x) * phi_i(x) ** 2,
                nodes[i - 1],
                nodes[i],
            )[0],
            integrate.quad(
                lambda x: p(x) * phi_i_minus_1_prime(x) * phi_i_prime(x)
                + q(x) * phi_i_minus_1(x) * phi_i(x),
                nodes[i - 1],
                nodes[i],
            )[0],
            integrate.quad(
                lambda x: -1 * f(x) * phi_i_minus_1(x), nodes[i - 1], nodes[i]
            )[0],
            integrate.quad(lambda x: -1 * f(x) * phi_i(x), nodes[i - 1], nodes[i])[0],
        ]
    return s


class BasisFunctionsArray:
    def __init__(self, nodes, *args, **kwargs) -> None:
        if isinstance(nodes, int):  # save amount of nodes
            self.n = nodes
        else:
            self.n = len(nodes)
        self.basisfunctions = basis_functions(nodes, *args, **kwargs)

    def __getitem__(self, index):
        return lambda x: self.basisfunctions(index, x)

    def __len__(self):
        # return self.n - 1  # one less basis function than nodes
        return self.n + 1  # one more basis function than nodes

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


if __name__ == "__main__":  # do not run when file is imported
    x_axis = np.linspace(0, 1, 1000)

    N = 10
    phi = basis_functions(N)
    print(calculate_S(N, phi))

    plt.subplot(121)

    for j in range(N + 1):
        plt.plot(x_axis, phi(j, x_axis))

    plt.subplot(122)
    phi = basis_functions([0, 0.1, 0.5, 0.6, 0.9, 1])
    print(calculate_S(5, phi))

    for j in range(6):
        plt.plot(x_axis, phi(j, x_axis))
    plt.show()

    # now with BasisFunctionsArray
    plt.subplot(121)
    basis_functions_array = BasisFunctionsArray(N)
    for f in basis_functions_array:
        plt.plot(x_axis, f(x_axis))

    plt.subplot(122)
    phi_0 = basis_functions_array[0]
    plt.plot(x_axis, phi_0(x_axis))

    plt.show()
