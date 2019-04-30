import networkx as nx

def degree(graph, k):
    return [x for x, y in sorted(graph.degree, key=lambda x: x[1], reverse=True)[:k]]

def single_degree_discount(graph, k):
    degree_count = dict(graph.degree)
    topk = []
    neighborhood_fn = graph.neighbors if isinstance(graph, nx.Graph) else graph.predecessors
    for _ in range(k):
        node = max(degree_count.items(), key=lambda x: x[1])[0]
        topk.append(node)
        for neighbor in neighborhood_fn(node):
            degree_count[neighbor] -= 1
    return topk

def top_k(graph, diffuse, k):
    expected_spread = []
    for node in graph.nodes:
        expected_spread.append([node, diffuse.diffuse_mc([node])])
    return [x for x, y in sorted(expected_spread, key=lambda x: x[1], reverse=True)[:k]]
