# %%
import networkx as nx
import matplotlib.pyplot as plt

# %%
G = nx.DiGraph()
G.add_edges_from(
    [
        ("DH", "S"),
        ("DH", "C"),
        ("DH", "W"),
        ("DH", "M"),
        ("DH", "T"),
        ("DH", "rS"),
        ("DFC", "LFC"),
        ("DFC", "C"),
        ("DFC", "G"),
        ("DFC", "I"),
        ("DFC", "WFC"),
        ("DFC", "PiFC"),
        ("DFC", "rL"),
        ("DFK", "LFK"),
        ("DFK", "I"),
        ("DFK", "WFK"),
        ("DFK", "PiFK"),
        ("DFK", "rL"),
        ("D", "DH"),
        ("D", "DFC"),
        ("D", "DFK"),
        ("S", "C"),
        ("LFC", "WFC"),
        ("LFC", "pK"),
        ("LFC", "I"),
        ("LFC", "C"),
        ("LFC", "G"),
        ("LFK", "WFK"),
        ("L", "LFC"),
        ("L", "LFK"),
        ("BB", "L"),
        ("BB", "D"),
        ("BB", "S"),
        ("BB", "Pi"),
        ("BB", "rS"),
        ("BB", "rL"),
        ("BB", "rB"),
        ("BG", "G"),
        ("BG", "M"),
        ("BG", "T"),
        ("BG", "rB"),
        ("KFC", "I"),
        ("KFK", "I"),
        ("KFK", "DK"),
        ("C", "CT"),
        ("C", "KFC"),
        ("C", "nC"),
        ("C", "beta"),
        ("C", "G"),
        ("G", "GT"),
        ("G", "KFC"),
        ("G", "nC"),
        ("G", "beta"),
        ("I", "IT"),
        ("I", "beta"),
        ("I", "DK"),
        ("DK", "DKT"),
        ("DK", "nK"),
        ("DK", "beta"),
        ("CT", "W"),
        ("CT", "M"),
        ("GT1", "Y"),
        ("GT1", "T"),
        ("GT1", "M"),
        ("GT1", "rB"),
        ("GT1", "BG"),
        ("IT", "CT"),
        ("IT", "GT"),
        ("IT", "beta"),
        ("IT", "pC"),
        ("DKT", "beta"),
        ("W", "omega"),
        ("WFC", "W"),
        ("WFC", "nC"),
        ("WFK", "omega"),
        ("WFK", "W"),
        ("WFK", "nK"),
        ("WFK", "nQ"),
        ("WFK", "omega"),
        ("M", "omega"),
        ("T1", "W"),
        ("T1", "C"),
        ("T1", "rS"),
        ("T1", "S"),
        ("PiFC1", "DFC"),
        ("PiFC1", "LFC"),
        ("PiFK1", "DFK"),
        ("PiFK1", "LFK"),
        ("Pi", "PiFC"),
        ("Pi", "PiFK"),
        ("rS", "rB"),
        ("rS", "Gamma"),
        ("rL", "rB"),
        ("rL", "Gamma"),
        ("rB", "psi"),
        ("rB", "omega"),
        ("rB", "u"),
        ("pC", "W"),
        ("pC", "omega"),
        ("pC", "beta"),
        ("pK", "WFK"),
        ("pK", "I"),
        ("beta", "nQ"),
        ("nCT", "CT"),
        ("nCT", "GT"),
        ("nCT", "pC"),
        ("nCT", "beta"),
        ("nKT", "DKT"),
        ("nKT", "beta"),
        ("nC", "nCT"),
        ("nC", "nKT"),
        ("nK", "nKT"),
        ("nK", "nCT"),
        ("nQ1", "nC"),
        ("nQ1", "nK"),
        ("nQ1", "pK"),
        ("nQ1", "I"),
        ("nQ1", "WFK"),
        ("nQ1", "omega"),
        ("nQ1", "W"),
        ("Y", "C"),
        ("Y", "G"),
        ("Y", "pK"),
        ("Y", "I"),
        ("omega1", "nC"),
        ("omega1", "nK"),
        ("omega1", "nQ"),
        ("psi", "pC"),
        ("g", "Y"),
        ("u", "nC"),
        ("u", "nK"),
    ]
)
G = G.reverse()

# %%
len(nx.recursive_simple_cycles(G))

# %%
nx.recursive_simple_cycles(G)
# %%
list(nx.topological_generations(G))
# %%
for layer, nodes in enumerate(nx.topological_generations(G)):
    # `multipartite_layout` expects the layer as a node attribute, so add the
    # numeric layer value as a node attribute
    for node in nodes:
        G.nodes[node]["layer"] = layer

# Compute the multipartite_layout using the "layer" node attribute
pos = nx.multipartite_layout(G, subset_key="layer")

with plt.style.context("default"):
    fig, ax = plt.subplots()
    nx.draw_networkx(G, pos=pos, ax=ax)
    ax.set_title("DAG layout in topological order")
    fig.tight_layout()
plt.show()
# %%
