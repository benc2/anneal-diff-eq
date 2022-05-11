import numpy as np
import matplotlib.pyplot as plt
import neal
from dwave.system import DWaveSampler, EmbeddingComposite
import imageio
import os
import webbrowser
from datetime import datetime
from main import simulated_sample, real_sample

from helper_functions import (
    Pi_functional,
    create_bqm,
    compute_a_min,
    compute_all_J_tildes,
)
from graph import show_bqm_graph
from basisfunctions import calculate_S


if __name__ == "__main__":
    N = 8
    r_min = 0.001
    r = 3
    # S = np.array([[1, 1, -2, 0, 0]] * N)
    # u_c = (
    #     np.array([-1 / N] + list(np.linspace(0, 1, N - 1) ** 0.7) + [np.e / N + 1]) + 1
    # )
    u_c = np.exp(np.linspace(0, 1, N + 1)) - 0.3
    S = calculate_S(N, p=1, q=1, f=0)  # S depends on the distance between the nodes
    H = 1
    J_hat = H  # set equal as in paper

    # here we need to do the embedding and stuff
    # Pi_min = Pi_functional(S, u_c)

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

    while r > r_min and ii < 20:
        J_tildes = compute_all_J_tildes(S, u_c, r)
        bqm = create_bqm(H, J_hat, J_tildes, boundary_condition="N")
        sampleset = simulated_sample(bqm, filter=False)
        # solver.sample(bqm)  # adjust this to the solver!
        a_min = compute_a_min(sampleset, u_c, r)
        print(Pi_functional(S, a_min), Pi_functional(S, u_c))
        if Pi_functional(S, a_min) < Pi_functional(S, u_c):
            u_c = a_min
        else:
            r /= 2
        # plt.subplot(0)
        # plt.subplots()
        plt.subplot(211)
        plt.title(f"Iteration: {ii}, r= {r}")
        x_axis = np.linspace(0, 1, 1000)
        plt.plot(x_axis, np.exp(x_axis))

        plt.plot(np.linspace(0, 1, N + 1), u_c)
        plt.subplot(212)
        # plt.title(f"Pi_min: {Pi_min}, new_Pi: {new_Pi}")

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
    imageio.mimsave(movie_filename, images, duration=0.5)
    for filename in filenames:
        os.remove(root_dir + filename)
    print("file://" + root_dir + movie_filename)
    webbrowser.open("file://" + root_dir + movie_filename)
