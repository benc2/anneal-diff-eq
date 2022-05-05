import numpy as np
import matplotlib.pyplot as plt


def tent(x, start, peak, end):
    if x <= start or x > end:
        # print(x, start, end)
        return 0
    elif start < x <= peak:
        return (x - start) / (peak - start)
    elif peak < x <= end:
        return 1 - (x - peak) / (end - peak)


def basisFunctions(nodes, shape="triangle", x_l=0, x_r=1):
    # give int to get equally spaced points, or give nodes directly
    if isinstance(nodes, int):
        nodes = np.linspace(x_l, x_r, N + 1)

    def phi_evaluation(i, x):
        return tent(x, nodes[i], nodes[i + 1], nodes[i + 2])

    phi_output = np.vectorize(
        phi_evaluation, otypes=[float]
    )  # makes it so you can input an array

    return phi_output


class BasisFunctionsArray:
    def __init__(self, nodes, *args, **kwargs) -> None:
        if isinstance(nodes, int):  # save amount of nodes
            self.n = nodes
        else:
            self.n = len(nodes)
        self.basisfunctions = basisFunctions(nodes, *args, **kwargs)

    def __getitem__(self, index):
        return lambda x: self.basisfunctions(index, x)

    def __iter__(self):
        for i in range(self.n - 1):  # one less basis function than nodes
            yield self[i]


if __name__ == "__main__":  # do not run when file is imported
    x_axis = np.linspace(0, 1, 1000)

    N = 10
    phi = basisFunctions(N)
    plt.subplot(121)

    for j in range(N - 1):
        plt.plot(x_axis, phi(j, x_axis))

    plt.subplot(122)
    phi = basisFunctions([0, 0.1, 0.5, 0.6, 0.9, 1])
    for j in range(4):
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
