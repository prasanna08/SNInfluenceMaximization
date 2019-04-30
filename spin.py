import numpy as np

def spin(graph, diffuse, k, perms):
    for node in graph.nodes:
        graph.nodes[node]['shapely'] = 0

    for i in range(perms):
        nodes = list(graph.nodes)
        np.random.shuffle(nodes)
        diffuse.shapely_diffuse(nodes)
        for node in nodes:
            graph.nodes[node]['shapely'] += diffuse.graph.nodes[node]['tmp']

    for node in graph.nodes:
        graph.nodes[node]['shapely'] /= float(perms)

    rank_list = sorted(((node, data['shapely']) for node, data in graph.nodes.data()), key=lambda x: x[1], reverse=True)
    topk_list = get_top_k_from_rank_list(rank_list, graph, k)
    return topk_list

def get_top_k_from_rank_list(rank_list, graph, k):
    topk = []
    removed_nodes = set()
    for node, shap in rank_list:
        if node not in removed_nodes:
            topk.append((node, shap))
            removed_nodes.union(set(graph.neighbors(node)))
        else:
            not_considered_nodes.append((node, shap))
        if len(topk) == k:
            return topk

    topk += not_considered_nodes[:k - len(topk)]
    return sorted(topk, key=lambda x: x[1], reverse=True)
