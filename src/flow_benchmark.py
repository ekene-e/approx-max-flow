from __future__ import division
import numpy as np
import networkx as nx
import sys
import time
import graph_util
from graph_util import EDGE_CAPACITY_ATTR
import sherman
from conductance_congestion_approx import ConductanceCongestionApprox
import sparsification

if len(sys.argv) != 6:
    print('usage: ' + sys.argv[0] + ' <networkx|sherman> <graph file> <source node list file> <sink node list file> <epsilon>')
    sys.exit(1)

algorithm = sys.argv[1]
graph_file = sys.argv[2]
source_file = sys.argv[3]
sink_file = sys.argv[4]
epsilon = float(sys.argv[5])

g = graph_util.deserialize_csv_adj_list(open(graph_file).read(), sep='\t')
for i in range(max(g.nodes())):
    g.add_node(i)
for u, v, c in graph_util.capacity_edge_iter(g):
    if c == 0.0:
        g.remove_edge(u, v)
sources = set(graph_util.deserialize_node_list(open(source_file).read()))
sinks = set(graph_util.deserialize_node_list(open(sink_file).read()))

print('n:', g.number_of_nodes())
print('m:', g.number_of_edges())

demands = np.array([(-1 if v in sources else (1 if v in sinks else 0)) for v in g.nodes()])

if algorithm == 'sherman':
    print('starting sherman')
    start_time = time.time()
    cong_approx = ConductanceCongestionApprox(g)
    sherman_flow = sherman.ShermanFlow(g, cong_approx)
    flow, flow_value = sherman_flow.max_flow(demands, epsilon)
    stop_time = time.time()
    print('sherman flow:\n', flow)
    print('sherman flow value:', flow_value)
    print('sherman time:', stop_time - start_time)
elif algorithm == 'networkx':
    g = g.to_undirected()
    super_source = max(g.nodes()) + 1
    g.add_node(super_source)
    super_sink = max(g.nodes()) + 1
    g.add_node(super_sink)
    for s in sources:
        g.add_edge(super_source, s)
    for s in sinks:
        g.add_edge(s, super_sink)
    print('starting networkx max flow')

    start_time = time.time()
    flow_val, flow = nx.maximum_flow(g, super_source, super_sink)
    stop_time = time.time()
    print('Networkx flow value:', flow_val)
    print('Networkx time:', stop_time - start_time)
else:
    print('Unknown algorithm: `{}`'.format(algorithm))
    sys.exit(1)

sys.exit(0)
