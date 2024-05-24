from __future__ import division
import numpy as np
import networkx as nx
import time
import matplotlib.pyplot as plt
from graph_util import set_edge_capacity, complete_graph
from mst_congestion_approx import MstCongestionApprox
from conductance_congestion_approx import ConductanceCongestionApprox
from sherman import ShermanFlow

def run_sherman_flow(g, cong_approx_class, source, sink, epsilon):
    cong_approx = cong_approx_class(g.to_undirected())
    sherman_flow = ShermanFlow(g, cong_approx)
    start_time = time.time()
    flow, flow_value = sherman_flow.max_st_flow(source, sink, epsilon)
    end_time = time.time()
    elapsed_time = end_time - start_time
    return elapsed_time, flow_value

def create_graph(num_vertices):
    g = complete_graph(num_vertices)
    for u, v, data in g.edges(data=True):
        data['capacity'] = np.random.uniform(1.0, 10.0)
    return g

def main():
    vertices_list = [10, 100, 1000]
    epsilon = 0.1
    source = 0
    sink = 1
    mst_times = []
    conductance_times = []

    for num_vertices in vertices_list:
        print(f"Running for {num_vertices} vertices...")
        g = create_graph(num_vertices)
        
        mst_time, _ = run_sherman_flow(g, MstCongestionApprox, source, sink, epsilon)
        mst_times.append(mst_time)
        
        conductance_time, _ = run_sherman_flow(g, ConductanceCongestionApprox, source, sink, epsilon)
        conductance_times.append(conductance_time)

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(vertices_list, mst_times, label="MST Congestion Approximation", marker='o')
    plt.plot(vertices_list, conductance_times, label="Conductance Congestion Approximation", marker='o')
    plt.xlabel("Number of Vertices")
    plt.ylabel("Time Taken (seconds)")
    plt.title("Time Comparison of Congestion Approximation Methods")
    plt.legend()
    plt.grid(True)
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

if __name__ == '__main__':
    main()
