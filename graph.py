import networkx as nx
import matplotlib.pyplot as plt
from dimod import to_networkx_graph
from helper_functions import parse_label


def show_bqm_graph(bqm):
    G = to_networkx_graph(bqm)
    node_positions = {}
    for node in G.nodes:
        k, i = parse_label(node)
        if k == 2:
            i -= 0.2
        node_positions[node] = (i, k)

    edge_colors = {}
    for node1, node2 in G.edges:
        if parse_label(node1)[1] == parse_label(node2)[1]:  # i.e. same i
            edge_colors[(node1, node2)] = "#F00"
        else:
            edge_colors[(node1, node2)] = "#000"

    # weights = [node[1]["bias"] for node in G.nodes(data=True)]
    nx.set_edge_attributes(G, edge_colors, "color")

    colors = nx.get_edge_attributes(G, "color").values()
    biases = list(nx.get_edge_attributes(G, "bias").values())
    edge_labels = {(n1, n2): round(d["bias"], 2) for n1, n2, d in G.edges(data=True)}

    latexed_node_labels = {
        node_label: "${}$".format(node_label) for node_label in G.nodes
    }
    # print(G.nodes(data=True))
    node_biases = {a: b for a, b in G.nodes(data="bias")}
    # G = nx.relabel_nodes(G, latexed_node_labels)
    nx.draw(
        G,
        pos=node_positions,
        width=biases,
        edge_color=colors,
    )

    node_positions_offset = {
        node: (position[0] + 0.1, position[1] + 0.1)
        for node, position in node_positions.items()
    }
    nx.draw_networkx_labels(G, pos=node_positions_offset, labels=latexed_node_labels)
    nx.draw_networkx_labels(G, pos=node_positions, labels=node_biases)
    nx.draw_networkx_edge_labels(
        G, node_positions, edge_labels=edge_labels, label_pos=0.4
    )
    plt.show()
