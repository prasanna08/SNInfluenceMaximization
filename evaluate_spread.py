from diffusion import TriggeringModel
import json
import networkx as nx
import spin

G = nx.read_gpickle('../Results/Co-Auth-2003.gpickle')
f = open('../Results/Shapely/final_shapely_tr.json')
data = json.load(f)
f.close()
rank_list = sorted(data.items(), key=lambda x: x[1], reverse=True)
diffuse = TriggeringModel(G)
expected_spreads = []
for i in [5, 10, 20, 40, 60, 100]:
	nodes = [n for n, v in spin.get_top_k_from_rank_list(rank_list, G, i)]
	spread = diffuse.diffuse_mc(nodes, mc=200)
	expected_spreads.append(spread)
print(expected_spreads)
