from congestion_approx import CongestionApprox
from graph_util import EDGE_CAPACITY_ATTR


class TreeCongestionApprox(CongestionApprox):
    def __init__(self, tree, tree_root, alpha):
        self.tree = tree.copy()
        self.root = tree_root
        self.cached_dfs_edges = list(self.recursive_dfs_edges(self.root, set(), False))
        self.cached_dfs_edges_data = list(self.recursive_dfs_edges(self.root, set(), True))
        self.alpha_upper = alpha

    def route_flow(self, demands):
        node_flow = dict(zip(self.tree.nodes(), demands))
        edge_flow = {}
        for parent, child in reversed(self.dfs_edges()):
            child_flow = node_flow[child]
            node_flow[parent] += child_flow
            edge_flow[(parent, child)] = child_flow
        return edge_flow

    def compute_node_potentials(self, edge_potentials):
        node_potentials = {self.root: 0}
        for edge, potential in zip(self.dfs_edges(), edge_potentials):
            parent, child = edge
            node_potentials[child] = node_potentials[parent] + potential
        return node_potentials

    def recursive_dfs_edges(self, cur_node, visited, data):
        if cur_node in visited:
            return
        visited.add(cur_node)
        if data:
            for neighbor, edict in self.tree[cur_node].items():
                if neighbor in visited:
                    continue
                yield (cur_node, neighbor, edict)
                yield from self.recursive_dfs_edges(neighbor, visited, data)
        else:
            for neighbor in self.tree[cur_node].keys():
                if neighbor in visited:
                    continue
                yield (cur_node, neighbor)
                yield from self.recursive_dfs_edges(neighbor, visited, data)

    def dfs_edges(self, data=False):
        if data:
            return self.cached_dfs_edges_data
        else:
            return self.cached_dfs_edges

    def compute_dot(self, b):
        flow = self.route_flow(b)
        return [flow[(u, v)] / edict[EDGE_CAPACITY_ATTR] / self.alpha() for (u, v, edict) in self.dfs_edges(data=True)]

    def compute_transpose_dot(self, x):
        edge_potentials = (xi / edict[EDGE_CAPACITY_ATTR] for (xi, (u, v, edict)) in zip(x, self.dfs_edges(data=True)))
        node_potentials = self.compute_node_potentials(edge_potentials)
        return [node_potentials[n] / self.alpha() for n in self.tree.nodes()]

    def alpha(self):
        return self.alpha_upper
