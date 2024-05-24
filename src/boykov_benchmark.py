from __future__ import division
import numpy as np
import graph_tool
import graph_tool.all as gt
import sys
import time

if len(sys.argv) != 2:
    print('usage: ' + sys.argv[0] + ' <n>')
    sys.exit(1)

n = int(sys.argv[1])
g = gt.Graph(directed=True)
g.add_vertex(n)
cap = g.new_edge_property("float")
for i in range(n):
    for j in range(n):
        if i == j:
            continue
        e = g.add_edge(i, j)
        cap[e] = 1

print('starting boykov')
start_time = time.time()
res = gt.boykov_kolmogorov_max_flow(g, g.vertex(0), g.vertex(1), cap)
res.a = cap.a - res.a
max_flow = sum(res[e] for e in g.vertex(1).in_edges())
stop_time = time.time()

print('flow:', max_flow)
print('time:', stop_time - start_time)
sys.exit(0)
