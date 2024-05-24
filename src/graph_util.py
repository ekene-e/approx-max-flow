from __future__ import division
import networkx as nx
import random

EDGE_CAPACITY_ATTR = 'capacity'

def get_edge_capacities(g):
    return [c for _, _, c in capacity_edge_iter(g)]

def get_edge_capacity(g, e):
    u, v = e
    return g[u][v][EDGE_CAPACITY_ATTR]

def set_edge_capacity(g, e, cap):
    u, v = e
    g[u][v][EDGE_CAPACITY_ATTR] = cap

def complete_graph(n):
    return diluted_complete_graph(n, 1.0)

def maximum_spanning_tree(g):
    g_ = g.to_undirected()
    for u, v, edict in g_.edges(data=True):
        edict[EDGE_CAPACITY_ATTR] = 1.0 / edict[EDGE_CAPACITY_ATTR]
    mst = nx.minimum_spanning_tree(g_, weight=EDGE_CAPACITY_ATTR)
    for u, v, edict in mst.edges(data=True):
        edict[EDGE_CAPACITY_ATTR] = 1.0 / edict[EDGE_CAPACITY_ATTR]
    return mst

def diluted_complete_graph(n, p):
    g = nx.DiGraph()
    g.add_nodes_from(range(n))
    g.add_edges_from([(i, j) for i in range(n) for j in range(i+1, n) if random.random() < p])
    for e in g.edges():
        set_edge_capacity(g, e, 1)
    return g

def edge_iter(g):
    for n, neighbor_dict in g.adjacency():
        for neighbor, _ in neighbor_dict.items():
            yield (n, neighbor)

def capacity_edge_iter(g):
    for n, neighbor_dict in g.adjacency():
        for neighbor, edge_data in neighbor_dict.items():
            yield (n, neighbor, edge_data[EDGE_CAPACITY_ATTR])

def cut_weight(g, vs):
    weight = 0
    for v in vs:
        adj_dict = g[v]
        for neighbor, data_dict in adj_dict.items():
            if neighbor not in vs:
                weight += data_dict[EDGE_CAPACITY_ATTR]
    return weight

def set_edge_weight(g, vs):
    weight = 0
    for v in vs:
        adj_dict = g[v]
        for neighbor, data_dict in adj_dict.items():
            if neighbor not in vs or neighbor < v:
                weight += data_dict[EDGE_CAPACITY_ATTR]
    return weight

def cut_conductance(g, vs):
    not_vs = set(g.nodes()) - vs
    min_s_edge_weight = min(set_edge_weight(g, vs), set_edge_weight(g, not_vs))
    if min_s_edge_weight == 0:
        return float("inf")
    else:
        return cut_weight(g, vs) / min_s_edge_weight

def estimate_conductance(g, n_samples):
    return min([cut_conductance(g, {v for v in g.nodes() if random.random() < 0.5}) for _ in range(n_samples)])

def deserialize_csv_adj_list(s, sep=','):
    g = nx.DiGraph()
    for line in s.splitlines():
        line = line.strip()
        row = line.split(sep)
        if len(row) < 2:
            print('warning: possibly malformed input line `{}`'.format(line))
            continue
        num_adj = int(row[1])
        if len(row) != (2 + 2 * num_adj):
            print('warning: possibly malformed input line `{}`'.format(line))
            continue
        node = int(row[0])
        for i in range(num_adj):
            neighbor_node = int(row[2 + 2 * i])
            edge_capacity = float(row[2 + 2 * i + 1])
            g.add_edge(node, neighbor_node, {EDGE_CAPACITY_ATTR: edge_capacity})
    return g

def serialize_csv_adj_list(g, sep=','):
    rows = []
    for u in g.nodes():
        neighbor_dict = g[u]
        num_neighbors = len(neighbor_dict)
        row = [u, num_neighbors]
        for v, edict in neighbor_dict.items():
            row.extend([v, edict[EDGE_CAPACITY_ATTR]])
        rows.append(row)
    return '\n'.join(sep.join(str(cell) for cell in row) for row in rows)

def deserialize_node_list(s):
    return [int(line.strip()) for line in s.splitlines() if line.strip()]

def serialize_node_list(ns):
    return '\n'.join(map(str, ns))

def gen_rand_2d_mesh(width, height):
    g = nx.DiGraph()
    for j in range(height):
        for i in range(width):
            cur_id = j * width + i
            x_nbr_id = j * width + (i + 1)
            y_nbr_id = (j + 1) * width + i
            if i < width - 1:
                g.add_edge(cur_id, x_nbr_id, {EDGE_CAPACITY_ATTR: random.random()})
            if j < height - 1:
                g.add_edge(cur_id, y_nbr_id, {EDGE_CAPACITY_ATTR: random.random()})
    return g

def gen_rand_3d_mesh(width, height, depth):
    g = nx.DiGraph()
    for k in range(depth):
        for j in range(height):
            for i in range(width):
                cur_id = k * width * height + j * width + i
                x_nbr_id = k * width * height + j * width + (i + 1)
                y_nbr_id = k * width * height + (j + 1) * width + i
                z_nbr_id = (k + 1) * width * height + j * width + i
                if i < width - 1:
                    g.add_edge(cur_id, x_nbr_id, {EDGE_CAPACITY_ATTR: random.random()})
                if j < height - 1:
                    g.add_edge(cur_id, y_nbr_id, {EDGE_CAPACITY_ATTR: random.random()})
                if k < depth - 1:
                    g.add_edge(cur_id, z_nbr_id, {EDGE_CAPACITY_ATTR: random.random()})
    return g

def cut_from_residuals(resid_g, source_vert):
    def dfs_on_resid_graph(curnode, visited):
        if curnode not in visited:
            visited.add(curnode)
            for neighbor in resid_g[curnode].keys():
                dfs_on_resid_graph(neighbor, visited)
    visited = set()
    dfs_on_resid_graph(source_vert, visited)

    cut_edges = set()
    for u, v in resid_g.edges():
        if u in visited and v not in visited:
            cut_edges.add((u, v))
        if v in visited and u not in visited:
            cut_edges.add((v, u))
    return cut_edges

def approx_min_cut_from_residuals(g, resid_map, source_vert, epsilon):
    resid_graph = g.reverse()
    for (u, v), resid in resid_map.items():
        if resid > epsilon:
            resid_graph.add_edge(u, v)
    return cut_from_residuals(resid_graph, source_vert)

def min_cut_from_residuals(g, resid_map, source_vert):
    return approx_min_cut_from_residuals(g, resid_map, source_vert, 0)

def multigraph_contract_edges(multi_g, es):
    if not es:
        return multi_g.copy()
    contracted_es = set((u, v) for (u, v, _) in es)
    nodes_old_to_new = {}
    meta_g = nx.MultiGraph(list(contracted_es))
    meta_g.add_nodes_from(multi_g.nodes())
    new_multi_g = nx.MultiGraph()

    # Each connected component in (V, es) is a node in the graph post-contraction
    for comp in nx.connected_components(meta_g):
        new_node = tuple(comp)
        new_multi_g.add_node(new_node)
        for n in comp:
            nodes_old_to_new[n] = new_node
    # Each edge in (E - es) is an edge in the graph post-contraction, provided
    # it is not a self-loop.
    for u, v, edict in multi_g.edges(data=True):
        if (u, v) in contracted_es or (v, u) in contracted_es:
            continue
        new_u = nodes_old_to_new[u]
        new_v = nodes_old_to_new[v]
        if new_u is new_v:
            continue
        new_multi_g.add_edge(new_u, new_v, **edict)
    return new_multi_g

def compute_mst(g):
    mst = nx.minimum_spanning_tree(g, weight=EDGE_CAPACITY_ATTR)
    return mst

def compute_mst_bottleneck_dist(g):
    def get_min_edge(g):
        if g.number_of_edges() > 0:
            return min(g.edges(data=True), key=lambda e: e[2][EDGE_CAPACITY_ATTR])
        else:
            return None

    def compute_mst_bottleneck_recursive(g, mst, bottleneck_dict):
        e = get_min_edge(mst)
        if not e:
            return
        u, v, data = e
        bottleneck_weight = data[EDGE_CAPACITY_ATTR]
        mst.remove_edge(u, v)
        sub_mst_A, sub_mst_B = nx.connected_component_subgraphs(mst)
        for u in sub_mst_A:
            for v in sub_mst_B:
                bottleneck_dict[(u, v)] = bottleneck_weight
                bottleneck_dict[(v, u)] = bottleneck_weight
        compute_mst_bottleneck_recursive(g, sub_mst_A, bottleneck_dict)
        compute_mst_bottleneck_recursive(g, sub_mst_B, bottleneck_dict)
        return

    bottleneck_dict = {}
    for comp in nx.connected_component_subgraphs(g):
        compute_mst_bottleneck_recursive(comp, compute_mst(comp), bottleneck_dict)
    return bottleneck_dict
