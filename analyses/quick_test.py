from __future__ import division
import numpy as np
import networkx as nx
import time
from graph_util import set_edge_capacity, complete_graph
from mst_congestion_approx import MstCongestionApprox
from conductance_congestion_approx import ConductanceCongestionApprox
from sherman import ShermanFlow

def main():
    num_vertices = 10
    epsilon = 0.1

    g = complete_graph(num_vertices)

    np.random.seed(42)  
    for u, v, data in g.edges(data=True):
        data['capacity'] = np.random.uniform(1.0, 10.0)

    # cong_approx = MstCongestionApprox(g.to_undirected())
    # alternatively can use ConductanceCongestionApprox:
    cong_approx = ConductanceCongestionApprox(g)

    sherman_flow = ShermanFlow(g, cong_approx)

    # run max flow
    source = 0
    sink = 1
    start_time = time.time()
    flow, flow_value = sherman_flow.max_st_flow(source, sink, epsilon)
    end_time = time.time()

    print(f"Max flow value: {flow_value}")
    print(f"Flow distribution: {flow}")
    print(f"Elapsed time: {end_time - start_time} seconds")

if __name__ == '__main__':
    main()
