import numpy as np
import matplotlib.pyplot as plt
from diff_eqn import DiffEqn

N = 6
r = 1.1 / N
r_min = 0.00001 * r
u_c = np.linspace(0, 1, N + 1) ** 2 * (1 - np.linspace(0, 1, N + 1))
diff_eq = DiffEqn(
    p=1,
    q=0,
    f=2,
    initial_condition=u_c,
    nodes=N,
    basis_functions="triangle",
    boundary_condition="D",
)  # ik heb ook "spline" als basis function toegevoegd maar dat ziet er niet zo mooi uit


# zo lost ie m op met het box algorithm
# je kan met de sampler parameter een sampler kiezen, als deze nog extra parameters heeft
# kan je die met sampler_config meegeven, als dict, bijv sampler_config = {"beta": [0.1, 4.2], "filter"=True}
solution = diff_eq.solve(r=r, r_min=r_min)
print(solution)
# je kan filename specificeren maar anders stopt ie m in de animations/datum folder
# hij heeft standaard plot opties, met graph aan of uit
diff_eq.animate(filename="animations/specialname.gif", graph=True)


# maar als je zelf wil bepalen hoe het eruit ziet kan je zo'n functie maken en als parameter meegeven
def plot_function(i):
    plt.title(f"Iteration {i}")
    x_axis = np.linspace(0, 1, 1000)
    plt.plot(x_axis, x_axis - x_axis**2)

    u_c = diff_eq.solution_iterates[i]
    a_min = diff_eq.a_min_iterates[i]
    plt.plot(np.linspace(0, 1, N + 1), u_c, "o")
    plt.plot(np.linspace(0, 1, N + 1), a_min, ".")
    plt.legend(["u_c", "a_min", "exact solution"])


diff_eq.animate(plot_function=plot_function)

# diff_eq.solution_function(u_c) returnt de functie f(x) = som u_c[i]*phi_i(x), handig als je
# basisfuncties hebt die geen tent zijn. diff_eq.animate gebruikt die standaard, tenzij je
# plot_function specificeert
