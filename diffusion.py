import numpy as np
import networkx as nx
from tqdm.autonotebook import tqdm

class TriggeringModel(object):
    def __init__(self, graph):
        """
        Args:
            graph: networkx graph object. Each node should have an attribute
                called node_dist which enumerates distribution over its neighbors.
        """
        self.graph = graph
        self.neighborhood_fn = self.graph.neighbors if isinstance(self.graph, nx.Graph) else self.graph.predecessors
        self.node_to_dist = {node: (len(d['node_dist']), [x[1] for x in d['node_dist']]) for node, d in self.graph.nodes.data()}
        self.node_to_sampled_dist = dict()

    def diffusion_iter(self):
        for node in self.graph.nodes:
            if self.graph.nodes[node]['is_active']:
                continue
            neighbors = self.neighborhood_fn(node)
            neighbors = [neighbor for neighbor in neighbors if self.graph.nodes[neighbor]['is_active'] and neighbor in self.graph.nodes[node]['activating_neighbors']]
            if len(neighbors) > 0:
                self.graph.nodes[node]['is_active'] = True
    
    def sample_node_dist_for_mc(self, mc):
        for node in self.graph.nodes:
            selected_nodes = np.random.multinomial(mc, pvals=self.node_to_dist[node][1])
            sampled_dist = []
            for selected_node, times in enumerate(selected_nodes):
                sampled_dist.extend([selected_node]*times)
            np.random.shuffle(sampled_dist)
            self.node_to_sampled_dist[node] = sampled_dist

    def sample_neighbors_for_nodes(self, mc_count):
        for node in self.graph.nodes:
            node_dist = self.graph.nodes[node]['node_dist']
            # node_dist = [(neighbors, prob) for all possible neighbors of node].
            # Select a set randomly from given distribution.
            selected_set = self.node_to_sampled_dist[node][mc_count]
            self.graph.nodes[node]['activating_neighbors'] = node_dist[selected_set][0]

    def diffuse(self, act_nodes, mc_count):
        self.sample_neighbors_for_nodes(mc_count)
        nx.set_node_attributes(self.graph, False, name='is_active')

        for node in act_nodes:
            self.graph.nodes[node]['is_active'] = True

        prev_active_nodes = set()
        active_nodes = set()
        while True:
            self.diffusion_iter()
            prev_active_nodes = active_nodes
            active_nodes = set(i for i, v in self.graph.nodes(data=True) if v['is_active'])
            if active_nodes == prev_active_nodes:
                break
        self.graph.total_activated_nodes.append(len(active_nodes))

    def diffuse_mc(self, act_nodes, mc=50):
        self.sample_node_dist_for_mc(mc)
        self.graph.total_activated_nodes = []
        for i in tqdm(range(mc), desc='Monte Carlo', leave=False):
            self.diffuse(act_nodes, i)
        return sum(self.graph.total_activated_nodes) / float(mc)

    def shapely_iter(self, act_nodes):
        nx.set_node_attributes(self.graph, False, name='is_active')

        for node in act_nodes:
            self.graph.nodes[node]['is_active'] = True

        self.diffusion_iter()
        active_nodes = [n for n, v in self.graph.nodes.data() if v['is_active']]
        return active_nodes

    def shapely_diffuse(self, nodes, mc=50):
        self.sample_node_dist_for_mc(mc)

        for node in nodes:
            self.graph.nodes[node]['tmp'] = 0

        for c in tqdm(range(mc), desc='Shapely Monte Carlo', leave=False):
            self.sample_neighbors_for_nodes(c)
            active_nodes_with = []
            active_nodes_without = []
            for i in range(len(nodes)):
                if i in active_nodes_with:
                    self.graph.nodes[node]['tmp'] = 0
                    continue
                active_nodes_with = self.shapely_iter(nodes[:i+1])
                active_nodes_without = self.shapely_iter(nodes[:i])
                self.graph.nodes[nodes[i]]['tmp'] +=  len(active_nodes_with) - len(active_nodes_without)

        for i in range(len(nodes)):
            self.graph.nodes[node]['tmp'] /= float(mc)

class IndependentCascade(object):
    def __init__(self, graph):
        self.graph = graph
        self.sampled_graph = graph.copy()
        self.edge_idx = {(u, v): i for i, (u, v) in enumerate(self.graph.edges())}
        self.reverse_edge_idx = {i: e for e, i in self.edge_idx.items()}
        self.prob_matrix = [self.graph.edges[self.reverse_edge_idx[i][0], self.reverse_edge_idx[i][1]]['prob'] for i in sorted(self.reverse_edge_idx.keys())]
    
    def sample_live_graph_mc(self, mc):
        edge_probs = {(u, v): d['prob'] for u, v, d in self.graph.edges().data()}
        probs = np.random.uniform(size=(mc, len(edge_probs)))
        self.sampled_graphs = []
        for p in probs:
            self.sampled_graphs.append(np.array([p > self.prob_matrix]).astype(np.int8))
        
    def sample_live_graph(self, mcount):
        removed_edges_idx = np.where(self.sampled_graphs[mcount] == 0)[1].tolist()
        removed_edges = [self.reverse_edge_idx[i] for i in removed_edges_idx]
        Gp = self.graph.copy()
        Gp.remove_edges_from(removed_edges)
        self.sampled_graph = Gp

    def diffusion_iter(self, act_nodes):
        new_act_nodes = set(act_nodes)
        for node in act_nodes:
            for node2 in nx.algorithms.bfs_tree(self.sampled_graph, node).nodes():
                new_act_nodes.add(node2)
        for node in new_act_nodes:
            self.sampled_graph.nodes[node]['is_active'] = True

    def diffuse(self, act_nodes, mcount):
        self.sample_live_graph(mcount)
        nx.set_node_attributes(self.sampled_graph, False, name='is_active')

        for node in act_nodes:
            self.sampled_graph.nodes[node]['is_active'] = True
        
        self.diffusion_iter(act_nodes)
        active_nodes = [n for n, v in self.sampled_graph.nodes.data() if v['is_active']]
        self.graph.total_activated_nodes.append(len(active_nodes))

    def diffuse_mc(self, act_nodes, mc=10):
        self.sample_live_graph_mc(mc)
        self.graph.total_activated_nodes = []
        for mcount in range(mc):
            self.diffuse(act_nodes, mcount)
        return sum(self.graph.total_activated_nodes) / float(mc)

    def shapely_iter(self, act_nodes):
        nx.set_node_attributes(self.sampled_graph, False, name='is_active')

        for node in act_nodes:
            self.sampled_graph.nodes[node]['is_active'] = True

        self.diffusion_iter(act_nodes)
        active_nodes = [n for n, v in self.sampled_graph.nodes.data() if v['is_active']]
        return active_nodes

    def shapely_diffuse(self, nodes, mc=10):
        self.sample_live_graph_mc(mc)
        for node in nodes:
            self.graph.nodes[node]['tmp'] = 0

        for c in tqdm(range(mc), desc='Shapely Monte Carlo', leave=False):
            self.sample_live_graph(c)
            active_nodes_with = []
            active_nodes_without = []
            for i in tqdm(range(len(nodes)), desc='Shapely Iter', leave=False):
                if i in active_nodes_with:
                    self.graph.nodes[node]['tmp'] = 0
                    continue
                active_nodes_with = self.shapely_iter(nodes[:i+1])
                active_nodes_without = self.shapely_iter(nodes[:i])
                self.graph.nodes[nodes[i]]['tmp'] +=  len(active_nodes_with) - len(active_nodes_without)

        for i in range(len(nodes)):
            self.graph.nodes[node]['tmp'] /= float(mc)

class LinearThreshold(object):
    def __init__(self, graph):
        self.graph = graph
        self.neighborhood_fn = self.graph.neighbors if isinstance(self.graph, nx.Graph) else self.graph.predecessors
    
    def sample_node_thresholds_mc(self, mc):
        self.sampled_thresholds = np.random.uniform(size=(mc, len(self.graph.nodes)))

    def sample_node_thresholds(self, mcount):
        for idx, node in enumerate(self.graph.nodes):
            self.graph.nodes[node]['threshold'] = self.sampled_thresholds[mcount][idx]

    def diffusion_iter(self):
        for node in self.graph.nodes:
            if self.graph.nodes[node]['is_active']:
                continue
            neighbors = self.neighborhood_fn(node)
            weights = sum(self.graph.edges[neighbor, node]['weight'] for neighbor in neighbors)
            if weights > self.graph.nodes[node]['threshold']:
                self.graph.nodes[node]['is_active'] = True

    def diffuse(self, act_nodes, mcount):
        self.sample_node_thresholds(mcount)
        nx.set_node_attributes(self.graph, False, name='is_active')

        for node in act_nodes:
            self.graph.nodes[node]['is_active'] = True

        prev_active_nodes = set()
        active_nodes = set()
        while True:
            self.diffusion_iter()
            prev_active_nodes = active_nodes
            active_nodes = set(i for i, v in self.graph.nodes(data=True) if v['is_active'])
            if active_nodes == prev_active_nodes:
                break
        self.graph.total_activated_nodes.append(len(active_nodes))

    def diffuse_mc(self, act_nodes, mc=50):
        self.sample_node_thresholds_mc(mc)
        self.graph.total_activated_nodes = []
        for mcount in range(mc):
            self.diffuse(act_nodes, mcount)
        return sum(self.graph.total_activated_nodes) / float(mc)

    def shapely_iter(self, act_nodes):
        nx.set_node_attributes(self.graph, False, name='is_active')

        for node in act_nodes:
            self.graph.nodes[node]['is_active'] = True

        self.diffusion_iter()
        active_nodes = [n for n, v in self.graph.nodes.data() if v['is_active']]
        return active_nodes

    def shapely_diffuse(self, nodes, mc=50):
        self.sample_node_thresholds_mc(mc)
        for node in nodes:
            self.graph.nodes[node]['tmp'] = 0

        for c in tqdm(range(mc), desc='Shapely Monte Carlo', leave=False):
            self.sample_node_thresholds(c)
            active_nodes_with = []
            active_nodes_without = []
            for i in tqdm(range(len(nodes)), desc='Shapely Nodes', leave=False):
                if i in active_nodes_with:
                    self.graph.nodes[node]['tmp'] = 0
                    continue
                active_nodes_with = self.shapely_iter(nodes[:i+1])
                active_nodes_without = self.shapely_iter(nodes[:i])
                self.graph.nodes[nodes[i]]['tmp'] +=  len(active_nodes_with) - len(active_nodes_without)

