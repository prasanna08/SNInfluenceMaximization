import utils
from collections import defaultdict

def build_iv_vector(graph, depth_limit=3):
    node_vectors = {}
    for node in graph.nodes:
        node_influence = defaultdict(int)
        node_influence[node] = 1
        for i in range(depth_limit):
            tmp = defaultdict(int)
            for node2 in node_influence:
                for node3 in graph.neighbors(node2):
                    if node2 == node3:
                        continue
                    tmp[node3] += 1/((i+1)**2) * node_influence[node2] * graph.edges[node2, node3]['prob']
            for node2 in tmp:
                node_influence[node2] += tmp[node2]
        node_vectors[node] = node_influence
    return node_vectors

def ivgreedy(graph, k, depth_limit=3):
    iv_vectors = build_iv_vector(graph, depth_limit=depth_limit)
    S = set()
    R = [(node, sum(v for k, v in iv_vectors[node].items())) for node in graph.nodes]
    max_node = max(R, key=lambda x: x[1])[0]
    S.add(max_node)
    A = set(graph.nodes)
    A.remove(max_node)
    AR = iv_vectors[max_node]

    for _ in range(1, k):
        g = []
        for node in A:
            gn = 0
            c = 1 - AR[node]
            for nnode in graph.nodes:
                p = c * iv_vectors[node][nnode]
                q = p + AR[nnode]
                p = p if q < 1 else 1 - AR[nnode]
                gn += p
            g.append((node, gn))
        max_node = max(g, key=lambda x: x[1])[0]
        S.add(max_node)
        A.remove(max_node)
        c = 1 - AR[max_node]
        for node in graph.nodes:
            AR[node] = min(1, AR[node] + c * iv_vectors[max_node][node])
    return S
