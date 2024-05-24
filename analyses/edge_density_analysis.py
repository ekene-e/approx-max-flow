import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import time
from mst_congestion_approx import MstCongestionApprox
from conductance_congestion_approx import ConductanceCongestionApprox
from sherman import ShermanFlow
from graph_util import set_edge_capacity

def run_sherman_flow(g, CongestionApproxClass, source, sink, epsilon):
    cong_approx = CongestionApproxClass(g.to_undirected())
    sherman_flow = ShermanFlow(g, cong_approx)
    start_time = time.time()
    flow, flow_value = sherman_flow.max_st_flow(source, sink, epsilon)
    end_time = time.time()
    return end_time - start_time, flow_value

def main():
    num_vertices = 5
    edge_densities = np.linspace(0.1, 1.0, 10) 
    epsilon = 0.1
    num_trials = 5

    mst_times = []
    conductance_times = []

    for density in edge_densities:
        mst_trial_times = []
        conductance_trial_times = []

        for _ in range(num_trials):
            g = nx.gnp_random_graph(num_vertices, density, directed=True)
            for u, v in g.edges():
                set_edge_capacity(g, (u, v), np.random.rand())

            source = 0
            sink = num_vertices - 1

            try:
                mst_time, _ = run_sherman_flow(g, MstCongestionApprox, source, sink, epsilon)
                conductance_time, _ = run_sherman_flow(g, ConductanceCongestionApprox, source, sink, epsilon)
            except Exception as e:
                print(f"Error occurred for density {density}: {e}")
                continue

            mst_trial_times.append(mst_time)
            conductance_trial_times.append(conductance_time)

        mst_times.append(np.mean(mst_trial_times))
        conductance_times.append(np.mean(conductance_trial_times))

    plt.figure(figsize=(10, 6))
    plt.plot(edge_densities, mst_times, 'b-x', label='MST Cong. Approx.')
    plt.plot(edge_densities, conductance_times, 'g-o', label='Conductance Cong. Approx.')
    plt.xlabel('Edge Density')
    plt.ylabel('Time Taken (seconds)')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.title('Time Complexity vs. Edge Density')
    plt.show()
    print("Edge Densities: ", edge_densities)
    print("MST Times: ", mst_times)
    print("Conductance Times: ", conductance_times)

if __name__ == '__main__':
    main()
