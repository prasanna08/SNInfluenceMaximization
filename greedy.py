from diffusion import TriggeringModel, IndependentCascade

def greedy(graph, diffuse, k):
	S = set()
	A = set(graph.nodes)
	while len(S) < k:
		node_diffusion = {}
		for node in A:
			S.add(node)
			node_diffusion[node] = diffuse.diffuse_mc(S)
			S.remove(node)
		max_node = max(node_diffusion.items(), key=lambda x: x[1])[0]
		S.add(max_node)
		A.remove(max_node)
	return S
