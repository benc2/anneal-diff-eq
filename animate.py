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
        if Sum != -1: # check whether v^i_1 + v^i_2 + v^i_1 == -1
            return False
    return True


if __name__ == "__main__":

    N = 2
    r_min = 0.05
    r = 1 / N
    # S = np.array([[1, 1, -2, 0, 0]] * N)

    u_c = np.array([0, 0.5, 1])
    # u_c= np.random.rand(N+1)
    u_c[0] = 0
    u_c[N] = 1
    # exit()
    S = calculate_S(N, p=1, q=0, f=0)  # S depends on the distance between the nodes
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

    while r > r_min and ii < 10000:
        J_tildes = compute_all_J_tildes(S, u_c, r)
        bqm = create_bqm(H, J_hat, J_tildes, boundary_condition="D")
        sampleset = simulated_sample(bqm)
        # solver.sample(bqm)  # adjust this to the solver!
        a_min = compute_a_min(sampleset, u_c, r)
        new_Pi = Pi_functional(S, a_min)
        print(Pi_min, new_Pi, r)
        if Pi_min < new_Pi:
            u_c = a_min
        else:
            r /= 2
            Pi_min = new_Pi
        # plt.subplot(0)
        # plt.subplots()
        plt.subplot(211)
        plt.title(f"Iteration: {ii}, r= {r}")

        plt.plot(np.linspace(0, 1, N + 1), u_c)
        plt.subplot(212)
        plt.title(f"Pi_min: {Pi_min}, new_Pi: {new_Pi}")

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
    imageio.mimsave(movie_filename, images, duration=0.005)
    for filename in filenames:
        os.remove(root_dir + filename)
    print("file://" + root_dir + movie_filename)
    webbrowser.open("file://" + root_dir + movie_filename)

    imageio.mimsave(f"movie_{time}.gif", images, duration=0.005)
    webbrowser.open(f"movie_{time}.gif")
