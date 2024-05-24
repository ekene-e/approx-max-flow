from __future__ import division
from graph_util import EDGE_CAPACITY_ATTR
from tree_congestion_approx import TreeCongestionApprox
import networkx as nx
import unittest
import numpy as np


class TreeCongestionApproxTest(unittest.TestCase):
    def test_dfs_edges(self):
        g = nx.Graph()
        g.add_edge('a', 'b')
        g.add_edge('b', 'c')
        g.add_edge('c', 'd')
        g.add_edge('c', 'e')
        tree_approx = TreeCongestionApprox(g, 'b', 1.0)

        dfs_edges = list(tree_approx.dfs_edges())
        self.assertLess(dfs_edges.index(('b', 'c')), dfs_edges.index(('c', 'd')))
        self.assertLess(dfs_edges.index(('b', 'c')), dfs_edges.index(('c', 'e')))
        for e in [('b', 'a'), ('b', 'c'), ('c', 'd'), ('c', 'e')]:
            self.assertIn(e, dfs_edges)
        self.assertEqual(len(dfs_edges), 4)

        for e in tree_approx.dfs_edges(data=True):
            u, v, edict = e
            self.assertIn((u, v), dfs_edges)
        self.assertEqual(len(dfs_edges), len(tree_approx.dfs_edges(data=True)))

    def test_compute_dot(self):
        g = nx.Graph()
        g.add_edge('a', 'b')
        g.add_edge('b', 'c')
        g.add_edge('c', 'd')
        g.add_edge('c', 'e')
        for u, v, edict in g.edges(data=True):
            edict[EDGE_CAPACITY_ATTR] = 2.5

        demands = {
            'a': -4,
            'b': 0,
            'c': 1,
            'd': 1,
            'e': 2,
        }
        b = [demands[n] for n in g.nodes()]
        tree_approx = TreeCongestionApprox(g, 'b', 1.0)

        Rb = tree_approx.compute_dot(b)
        expected_Rb = {
            ('b', 'a'): -4 / 2.5,
            ('b', 'c'): 4 / 2.5,
            ('c', 'd'): 1 / 2.5,
            ('c', 'e'): 2 / 2.5,
        }
        for e, Rbi in zip(tree_approx.dfs_edges(), Rb):
            self.assertEqual(expected_Rb[e], Rbi)
        self.assertEqual(len(g.edges()), len(Rb))

    def test_compute_transpose_dot(self):
        g = nx.Graph()
        g.add_edge('a', 'b')
        g.add_edge('b', 'c')
        g.add_edge('c', 'd')
        g.add_edge('c', 'e')
        for u, v, edict in g.edges(data=True):
            edict[EDGE_CAPACITY_ATTR] = 2.5

        t = g
        root = 'b'
        tree_approx = TreeCongestionApprox(t, root, 1.0)

        x = [1, 2, 3, 4]
        r_transpose_x = tree_approx.compute_transpose_dot(x)
        for i in range(5):
            e_i_hat = [0, 0, 0, 0, 0]
            e_i_hat[i] = 1
            r_e_i_hat = tree_approx.compute_dot(e_i_hat)
            self.assertEqual(r_transpose_x[i], np.dot(r_e_i_hat, x))


if __name__ == '__main__':
    unittest.main()
