from __future__ import division
import numpy as np
import sys
import time
import graph_util
import sherman
from mst_congestion_approx import MstCongestionApprox

if len(sys.argv) != 3:
    print('usage: {} <num vertices> <epsilon>'.format(sys.argv[0]))
    sys.exit(1)

n = int(sys.argv[1])
epsilon = float(sys.argv[2])
print('{}-approximate max-flow on {}-complete graph\n'.format(epsilon, n))

g = graph_util.complete_graph(n)
print('n:', n)
print('m:', g.number_of_edges())

cong_approx = MstCongestionApprox(g.to_undirected())
sherman_flow = sherman.ShermanFlow(g, cong_approx)

start_time = time.time()
flow, _ = sherman_flow.max_st_flow(0, 1, epsilon)
stop_time = time.time()

print('final flow:\n', flow)
print('time:', stop_time - start_time)
sys.exit(0)
