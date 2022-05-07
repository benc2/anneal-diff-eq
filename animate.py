import numpy as np

from main import *
import imageio
from datetime import datetime
import webbrowser


if __name__ == "__main__":

    N = 3
    r_min = 0.02
    r = 0.5
    S = np.array([[1, 1, -2, 0, 0]] * N)
    u_c = np.array(np.linspace(0,1,N+1))
    H = 1
    J_hat = H  # set equal as in paper

    # here we need to do the embedding and stuff
    solver = None
    Pi_min = Pi_functional(S, u_c)

    # box algorithm
    ii = 0
    filenames = []
    plt.figure(figsize=(8, 6), dpi=80)

    while r > r_min and ii<10000:
        J_tildes = compute_all_J_tildes(S, u_c, r)
        bqm = create_bqm(H, J_hat, J_tildes, boundary_condition="D")
        sampleset = simulated_sample(bqm)
        # solver.sample(bqm)  # adjust this to the solver!
        a_min = compute_a_min(sampleset, u_c, r)
        new_Pi = Pi_functional(S, a_min)
        print(Pi_min,new_Pi,r)
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

        show_bqm_graph(bqm,False)
        # plt.subplot(1)
        plt.savefig(f"out{ii}.png")
        filenames.append(f"out{ii}.png")
        # plt.show()
        ii+=1
        plt.clf()
    # plt.show()
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    now = datetime.now()  # current date and time

    time = now.strftime("%H.%M.%S")

    imageio.mimsave(f"movie_{time}.gif", images, duration=0.005)
    webbrowser.open(f"movie_{time}.gif")