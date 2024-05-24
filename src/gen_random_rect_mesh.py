import graph_util
import networkx as nx
import sys

if len(sys.argv) != 3 and len(sys.argv) != 4:
    print('usage: {} <width> <height> [<depth>]'.format(sys.argv[0]))
    sys.exit(1)

width = int(sys.argv[1])
height = int(sys.argv[2])
depth = int(sys.argv[3]) if len(sys.argv) == 4 else None

if depth:
    g = graph_util.gen_rand_3d_mesh(width, height, depth)
else:
    g = graph_util.gen_rand_2d_mesh(width, height)

print(graph_util.serialize_csv_adj_list(g, sep='\t'))
