from __future__ import division

import graph_util
import sparsification
import networkx
import random
import sys
import time

if len(sys.argv) != 7:
    print('usage: %s <num trials> <num vertices> <num edges> <min capacity> <max capacity> <epsilon>' % sys.argv[0])
    sys.exit(1)

n_trials = int(sys.argv[1])
n = int(sys.argv[2])
m = int(sys.argv[3])
max_cap = float(sys.argv[4])
min_cap = float(sys.argv[5])
epsilon = float(sys.argv[6])
g = networkx.Graph()
all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
random.shuffle(all_edges)
g.add_edges_from(all_edges[:m])
for e in g.edges():
    cap = random.random() * (max_cap - min_cap) + min_cap
    graph_util.set_edge_capacity(g, e, cap)

min_cut_actual = networkx.minimum_cut(g, 0, 1, capacity='capacity')[0]

total_elapsed = 0
for i in range(n_trials):
    print('trial: %d' % i)
    t_start = time.time()
    if max_cap == min_cap:
        sparse_g = sparsification.sparsify(g, epsilon)
    else:
        sparse_g = sparsification.weighted_sparsify(g, epsilon)
    elapsed_time = time.time() - t_start
    total_elapsed += elapsed_time
    min_cut_sparsed = networkx.minimum_cut(sparse_g, 0, 1, capacity='capacity')[0]
    error = abs(min_cut_sparsed - min_cut_actual) / abs(min_cut_actual)
    print('|E sparse| / |E|: %f' % (sparse_g.number_of_edges() / g.number_of_edges()))
    print('relative error: %f' % error)
    print('elapsed time: %f' % elapsed_time)

print('================')
print('elapsed time, average: %f' % (total_elapsed / n_trials))
