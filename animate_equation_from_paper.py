import numpy as np

from main import *
import imageio
from datetime import datetime
import webbrowser
import os
from helper_functions import compute_J_tilde
from basisfunctions import calculate_S
from helper_functions import parse_label


def check_admissibility(sampleset):
    sample = sampleset.first.sample
    N = len(sample) // 3 - 1  # remember there are N+1 nodes
    qubit_array = np.zeros((N + 1, 3))
    for label, value in sample.items():
        k, i = parse_label(label)
        qubit_array[i, k - 1] = value

    for i in range(N):
        Sum = sum(qubit_array[i])
        if Sum != -1:  # check whether v^i_1 + v^i_2 + v^i_1 == -1
            return False
    return True


if __name__ == "__main__":

    N = 10

    r = 0.06
    r_min = 0.00001 * r
    # S = np.array([[1, 1, -2, 0, 0]] * N)



    # S matrix specific for the equation
    # See doi:10.1155/2012/180806 example 3.1
    alpha = 4
    beta = -0.8
    gamma = 2.5
    p = lambda x: alpha * (1+ beta * x)**gamma
    q= lambda x: - alpha * beta**2 *gamma * (gamma-2) / 4 * (1+ beta * x)**(gamma-2)
    nodes = np.linspace(0,1,N+1)**0.5
    # S = calculate_S(N, p=p, q=q, f=lambda x: -2*np.sin(a*x)/(a*x**3) + 2*np.cos(a* x)/(x**2) + a* np.sin(a*x)/x if x!= 0 else a**2/3)
    S = calculate_S(nodes, p=p, q=q, f=0)
    solution = lambda x: 1/np.sqrt(p(x)) *x*np.sqrt(p(1)) # exact solution

    vec = np.linspace(0,1,N+1)
    vec = (vec*(1-vec))**0.4
    # define an initial state close to the solution
    u_c = np.array([solution(x) for x in np.linspace(0, 1, N + 1)]) + vec * (np.random.rand(N+1))
    print(u_c)
    # apply boundary conditions to initial state
    u_c[0] = 0
    u_c[N] = 1


    H = 1
    J_hat = H  # set equal as in paper

    # here we need to do the embedding and stuff
    solver = None
    Pi_min = Pi_functional(S, u_c)

    # box algorithm
    ii = 0
    filenames = []
    plt.figure(figsize=(8, 6), dpi=80)

    now = datetime.now()  # current date and time
    time = now.strftime("%H.%M.%S")
    date = now.strftime(r"%d-%m-%Y")
    folder = f"animations/{date}/"
    # folder = ""
    root_dir = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/") + "/"
    try:
        os.makedirs(root_dir + folder)
    except FileExistsError:
        pass
    old_values = 0
    counter = 0
    while r > r_min and counter < 4 and ii < 100:
        print(ii)
        J_tildes = compute_all_J_tildes(S, u_c, r)
        bqm = create_bqm(H, J_hat, J_tildes, boundary_condition="D")
        sampleset = simulated_sample(bqm)
        # solver.sample(bqm)  # adjust this to the solver!

        a_min = compute_a_min(sampleset, u_c, r)

        if Pi_functional(S, a_min) < Pi_functional(S, u_c):
            u_c = a_min
        else:
            r /= 2
        # plt.subplot(0)
        # plt.subplots()
        plt.subplot(211)
        plt.title(f"Iteration: {ii}, r= {r}")

        plt.plot(nodes, u_c, "o")
        plt.plot(nodes, a_min, ".")
        plt.plot(np.linspace(0, 1, 1001), [solution(x) for x in np.linspace(0, 1, 1001)])
        plt.legend(["u_c", "a_min", "exact solution"])
        plt.subplot(212)

        show_bqm_graph(bqm, False)
        # plt.subplot(1)
        filename = folder + f"out{ii}.png"
        plt.savefig(filename)
        filenames.append(filename)
        # plt.show()
        ii += 1
        plt.clf()
    # plt.show()
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    now = datetime.now()  # current date and time

    movie_filename = folder + f"movie_{time}.gif"
    imageio.mimsave(movie_filename, images, duration=0.1)
    for filename in filenames:
        os.remove(root_dir + filename)
    print("file://" + root_dir + movie_filename)
    webbrowser.open("file://" + root_dir + movie_filename)

    # imageio.mimsave(f"movie_{time}.gif", images, duration=0.005)
    # webbrowser.open(f"movie_{time}.gif")
