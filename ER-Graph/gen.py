import sys
import os
import random
import networkx as nx


def connect_isolated_nodes(G, max_edges=2, seed=None):
    rng = random.Random(seed)

    isolates = list(nx.isolates(G))
    nodes = list(G.nodes())

    for u in isolates:
        # choose 1 or 2 connections
        k = rng.randint(1, max_edges)

        # candidate nodes (exclude itself)
        candidates = [v for v in nodes if v != u]

        # sample without replacement
        targets = rng.sample(candidates, k)

        for v in targets:
            G.add_edge(u, v)

    return G

def generate_graph(n, avg_degree, seed, filepath):
    p = avg_degree / (n - 1)
    # while True:
    #     G = nx.erdos_renyi_graph(n=n, p=p, seed=seed)
    #     if min(dict(G.degree()).values()) > 0:
    #         break
    G = nx.erdos_renyi_graph(n=n, p=p, seed=seed)
    G = connect_isolated_nodes(G, max_edges=2, seed=seed)
    num_isolates = nx.number_of_isolates(G)
    print(f"Number of isolated nodes: {num_isolates}")

    with open(filepath, "w") as f:
        for u, v in G.edges():
            f.write(f"{float(u):.10f} {float(v):.10f}\n")

def generate_anchor_edges(n, num_edges, out_dir):
    if num_edges > n:
        raise ValueError("Number of anchor edges cannot exceed number of nodes")

    nodes = list(range(n))
    random.shuffle(nodes)

    anchors = nodes[:num_edges]

    random.shuffle(anchors)
    split_idx = int(0.2 * num_edges)

    anchors_20 = anchors[:split_idx]
    anchors_80 = anchors[split_idx:]

    with open(os.path.join(out_dir, "anchor_links.txt"), "w") as f:
        for v in anchors_20:
            f.write(f"{float(v):.10f} {float(v):.10f}\n")

    with open(os.path.join(out_dir, "test_pairs.txt"), "w") as f:
        for v in anchors_80:
            f.write(f"{float(v):.10f} {float(v):.10f}\n")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print(
            "Usage: python gen_graphs.py "
            "<num_nodes> <avg_degree> <output_dir> <num_anchor_edges>"
        )
        sys.exit(1)

    n = int(sys.argv[1])
    avg_degree = float(sys.argv[2])
    out_dir = sys.argv[3]
    num_anchor_edges = int(sys.argv[4])

    os.makedirs(out_dir, exist_ok=True)

    generate_graph(n, avg_degree, seed=1,
                   filepath=os.path.join(out_dir, "graph1.txt"))
	
    generate_graph(n, avg_degree, seed=2,
                   filepath=os.path.join(out_dir, "graph2.txt"))

	
    generate_anchor_edges(n, num_anchor_edges, out_dir)
