import random
import networkx as nx
from functools import reduce

def sample_random_distribution(size):
    """
    For more info on this: http://www.cs.cmu.edu/~nasmith/papers/smith+tromble.tr04.pdf
    """
    x = [0] + sorted([random.random() for i in range(size-1)]) + [1]
    dist = [i - j for i, j in zip(x[1:], x[:-1])]
    return dist

def create_dist_for_graph(graph):
    for node in graph.nodes:
        neighbors = list(graph.neighbors(node))
        n = len(neighbors)
        dist = sample_random_distribution(2**n)
        node_dist = []
        for i, d in enumerate(dist):
            nset = [neighbors[j] for j, k in enumerate(format(i, '#0%db' % (n+2))[2:]) if k == '1']
            node_dist.append((nset, d))
        graph.nodes[node]['node_dist'] = node_dist
    return graph

def create_dist_for_graph_edges(graph):
    probs = [random.random() for _ in range(len(graph.edges))]
    for edge, p in zip(graph.edges, probs):
        graph.edges[edge]['prob'] = p
    return graph

def create_normalized_weights_for_graph_edges(graph):
    graph = create_dist_for_graph_edges(graph)
    graph = nx.DiGraph(graph)
    for node in graph.nodes:
        for edge in graph.in_edges(node):
            graph.edges[edge]['weight'] = graph.edges[edge]['prob'] / graph.in_degree(node)
    return graph

def generate_edge_weights_from_dist(graph):
    for node in graph.nodes:
        for neighbor in graph.neighbors(node):
            p = sum(dist[1] for dist in graph.nodes[node]['node_dist'] if neighbor in dist)
            graph.edges[neighbor, node]['prob'] = p
    return graph

def generate_union_node_dist_from_graph(graph):
    for node in graph.nodes:
        neighbors = list(graph.neighbors(node))
        n = len(neighbors)
        print("Createing for node %d with %d neighbors" % (node, n))
        node_dist = []
        total_weight = 0.0
        for i in range(2**n):
            print('Generating iter %d' % i)
            nset = [neighbors[j] for j, k in enumerate(format(i, '#0%db' % (n+2))[2:]) if k == '1']
            nnset = set()
            if nset:
                nnset = reduce(lambda x, y: x.union(y), [set(graph.neighbors(nsetnode)) for nsetnode in nset])
            node_dist.append((nset, len(nnset) if nnset else 0.0))
            total_weight += len(nnset) if nnset else 0.0
        if total_weight > 0.0:
            node_dist = [(nset, weight/total_weight) for nset, weight in node_dist]
        graph.nodes[node]['node_dist'] = node_dist
    return graph

def generate_intersection_node_dist_from_graph(graph):
    for node in graph.nodes:
        neighbors = list(graph.neighbors(node))
        n = len(neighbors)
        node_dist = []
        total_weight = 0.0
        for i in range(2**n):
            nset = [neighbors[j] for j, k in enumerate(format(i, '#0%db' % (n+2))[2:]) if k == '1']
            nnset = set()
            if nset:
                nnset = reduce(lambda x, y: x.intersection(y), [set(graph.neighbors(nsetnode)) for nsetnode in nset])
            node_dist.append((nset, len(nnset) if nnset else 0.0))
            total_weight += len(nnset) if nnset else 0.0
        if total_weight > 0.0:
            node_dist = [(nset, weight/total_weight) for nset, weight in node_dist]
        graph.nodes[node]['node_dist'] = node_dist
    return graph

def generate_weighted_union_node_dist_from_graph(graph):
    for node in graph.nodes:
        neighbors = list(graph.neighbors(node))
        n = len(neighbors)
        node_dist = []
        total_weight = 0.0
        for i in range(2**n):
            nset = [neighbors[j] for j, k in enumerate(format(i, '#0%db' % (n+2))[2:]) if k == '1']
            successor_sum = sum(graph.edges[nsetnode, neighbornset]['weight'] for nsetnode in nset for neighbornset in graph.successor(nsetnode))
            node_sum = sum(graph.edges[nsetnode, node] for nsetnode in nset)
            node_dist.append((nset, float(node_sum)/successor_sum))
            total_weight += float(node_sum)/successor_sum
        if total_weight > 0.0:
            node_dist = [(nset, weight/total_weight) for nset, weight in node_dist]
        graph.nodes[node]['node_dist'] = node_dist
    return graph

def generate_weighted_intersection_node_dist_from_graph(graph):
    for node in graph.nodes:
        neighbors = list(graph.neighbors(node))
        n = len(neighbors)
        node_dist = []
        total_weight = 0.0
        for i in enumerate(2**n):
            nset = [neighbors[j] for j, k in enumerate(format(i, '#0%db' % (n+2))[2:]) if k == '1']
            nnset = reduce(lambda x, y: x.intersection(y), [set(graph.neighbors(nsetnode)) for nsetnode in nset])
            successor_sum = sum(graph.edges[nsetnode, neighbornset]['weight'] for neighbornset in nnset for nsetnode in nset)
            node_sum = sum(graph.edges[nsetnode, node] for nsetnode in nset)
            node_dist.append((nset, float(node_sum)/successor_sum))
            total_weight += float(node_sum)/successor_sum
        if total_weight > 0.0:
            node_dist = [(nset, weight/total_weight) for nset, weight in node_dist]
        graph.nodes[node]['node_dist'] = node_dist
    return graph

def parse_graph_txt_file(fname, separator='\t'):
    edge_list = []
    with open(fname, 'r') as f:
        for line in f.readlines():
            if line.startswith('#'):
                continue
            edge_list.append(list(map(int, line.strip().split(separator))))
    G = nx.Graph()
    G.add_edges_from(edge_list)
    return G, edge_list