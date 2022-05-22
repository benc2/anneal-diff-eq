import copy
import os, shutil
from datetime import datetime

import matplotlib.pyplot as plt
import imageio
import neal
import numpy as np
import webbrowser
from dwave.system import DWaveSampler, EmbeddingComposite

from basisfunctions import BasisFunctionsArray, calculate_S
from graph import show_bqm_graph
from helper_functions import (
    Pi_functional,
    compute_a_min,
    compute_all_J_tildes,
    create_bqm,
    feasible_solution,
)


def simulated_sample(bqm, **kwargs):
    sim_solver = neal.SimulatedAnnealingSampler()
    sampleset = sim_solver.sample(bqm, num_reads=1000, **kwargs).aggregate()
    return sampleset


def real_sample(bqm, **kwargs):
    real_solver = EmbeddingComposite(DWaveSampler())
    return real_solver.sample(bqm, **kwargs).aggregate()


def sample(bqm, sampler, filter=False, **kwargs):
    if not filter:
        return sampler.sample(bqm, **kwargs).aggregate()
    else:
        while True:
            sampleset = sampler.sample(bqm, **kwargs).filter(feasible_solution)
            if len(sampleset) > 0:
                return sampleset


class DiffEqn:
    def __init__(
        self,
        p,
        q,
        f,
        initial_condition,
        nodes=None,
        x_l=0,
        x_r=1,
        boundary_condition="D",
        basis_functions="triangle",
    ) -> None:
        self.boundary_condition = boundary_condition
        self.initial_condition = initial_condition
        self.basis_functions_shape = basis_functions
        self.N = len(initial_condition) - 1

        if isinstance(nodes, int):
            self.N = nodes
            self.nodes = np.linspace(x_l, x_r, nodes + 1)

        else:
            self.nodes = nodes
            if len(self.initial_condition) != len(self.nodes):
                raise ValueError(
                    f"Lengths of nodes ({len(nodes)}) and initial condition ({len(self.initial_condition)}) do not match"
                )

        self.S = calculate_S(nodes, self.basis_functions_shape, p=p, q=q, f=f)

        # currently not compatible with how calculate_S works, it should accept any basis functions
        self.solution = None
        self.solution_iterates = []
        self.bqm_iterates = []
        self.a_min_iterates = []

    # def create_bqm(self, b_c_strength=1):
    #     self.bqm = helper_create_bqm()
    #     pass

    def solution_function(self, coefficients):
        basis_functions = BasisFunctionsArray(self.nodes, self.basis_functions_shape)

        @np.vectorize
        def fct(x):
            total = 0
            for coeff, basis_fct in zip(coefficients, basis_functions):
                total += coeff * basis_fct(x)
            return total

        return fct

    def plot_solution(self, coefficients, **kwargs):
        x_axis = np.linspace(self.nodes[0], self.nodes[-1], 1000)
        sol_fct = self.solution_function(coefficients)
        plt.plot(x_axis, sol_fct(x_axis), **kwargs)

    def default_plot(self, i):
        u_c = self.solution_iterates[i]
        self.plot_solution(u_c)
        plt.title(f"Iteration {i}")

    def plot_with_graph(self, i):
        u_c = self.solution_iterates[i]
        bqm = self.bqm_iterates[i]
        r = self.r_iterates[i]
        plt.subplot(211)
        self.plot_solution(u_c)
        plt.title(f"Iteration {i}")
        plt.subplot(212)
        show_bqm_graph(bqm, show=False)
        plt.tight_layout()

    def solve(
        self,
        r,
        r_min=None,
        Pi_min=None,
        sampler=None,
        H=1,
        J_hat=1,
        b_c_strength=1,
        sampler_config=None,
    ):
        if sampler is None:
            sampler = simulated_sample

        if sampler_config is None:
            sampler_config = {}

        if (r_min is None) == (Pi_min is None):
            raise ValueError("Set either r_min or Pi_min for the stopping condition")

        u_c = copy.copy(self.initial_condition)
        self.solution_iterates = []
        self.r_iterates = []
        self.bqm_iterates = []
        self.a_min_iterates = []

        while r > r_min:
            J_tildes = compute_all_J_tildes(self.S, u_c, r)
            bqm = create_bqm(
                H,
                J_hat,
                J_tildes,
                boundary_condition=self.boundary_condition,
                b_c_strength=b_c_strength,
            )
            self.bqm_iterates.append(bqm)
            sampleset = sampler(bqm, **sampler_config)
            # solver.sample(bqm)  # adjust this to the solver!
            a_min = compute_a_min(sampleset, u_c, r)
            self.a_min_iterates.append(a_min)
            self.r_iterates.append(r)

            if Pi_functional(self.S, a_min) < Pi_functional(self.S, u_c):
                u_c = a_min
            else:
                r /= 2

            self.solution_iterates.append(u_c)

        return u_c

    def animate(
        self,
        filename=None,
        preview=True,
        graph=False,
        duration=0.05,
        plot_function=None,
        **kwargs,
    ):
        if plot_function is None:
            if graph:
                plot_function = self.plot_with_graph
            else:
                plot_function = self.default_plot
        # for now without graph
        if filename:
            folder = os.path.dirname(filename).replace("\\", "/") + "/"
            file_base = os.path.basename(filename)

        else:
            now = datetime.now()  # current date and time
            time = now.strftime("%H.%M.%S")
            date = now.strftime(r"%d-%m-%Y")
            folder = f"animations/{date}/"
            file_base = f"movie_{time}.gif"

        root_dir = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/") + "/"
        absolute_folder = root_dir + folder
        frame_folder = absolute_folder + "frames/"
        try:
            os.makedirs(frame_folder)
        except FileExistsError:
            pass

        absolute_filename = absolute_folder + file_base

        # fig, ax = plt.subplots()

        images = []
        for i in range(len(self.solution_iterates)):
            plot_function(i)
            # self.plot_solution(u_c, **kwargs)
            # plt.title(f"Iteration {i}")
            plt.savefig(frame_folder + f"frame_{i}.png")
            images.append(imageio.imread(frame_folder + f"frame_{i}.png"))
            plt.clf()

        shutil.rmtree(frame_folder)
        imageio.mimsave(absolute_filename, images, duration=duration)

        if preview:
            webbrowser.open("file://" + absolute_filename)
