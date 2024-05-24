from __future__ import division
import graph_util
import networkx as nx
import numpy as np
import numpy.testing as npt
import sherman
from conductance_congestion_approx import ConductanceCongestionApprox
import unittest
from mst_congestion_approx import MstCongestionApprox


class ShermanTest(unittest.TestCase):
    def test_compute_C(self):
        g = graph_util.complete_graph(5)
        graph_util.set_edge_capacity(g, (0, 1), 12)
        graph_util.set_edge_capacity(g, (0, 2), 13)
        congestion_approximator = ConductanceCongestionApprox(g)
        sherman_flow = sherman.ShermanFlow(g, congestion_approximator)

        x = np.ones(g.number_of_edges())
        x[0] = 2
        x[1] = 3
        expected = np.ones(g.number_of_edges())
        expected[0] = 2 * 12
        expected[1] = 3 * 13
        npt.assert_array_equal(expected, sherman_flow.compute_C(x))

        expected[0] = 2 / 12
        expected[1] = 3 / 13
        npt.assert_array_equal(expected, sherman_flow.compute_Cinv(x))

    def test_compute_B(self):
        g = graph_util.complete_graph(5)
        congestion_approximator = ConductanceCongestionApprox(g)
        sherman_flow = sherman.ShermanFlow(g, congestion_approximator)

        x = np.zeros(g.number_of_edges())
        x[0] = 2
        x[1] = -1.2
        x[2] = -0.1
        expected = np.zeros(g.number_of_nodes())
        expected[0] = -2 + 1.2 + 0.1
        expected[1] = 2
        expected[2] = -1.2
        expected[3] = -0.1
        actual_Bx = sherman_flow.compute_B(x)
        npt.assert_array_equal(expected, actual_Bx)

        # Test BT
        b = np.zeros(g.number_of_nodes())
        b[0] = 1
        b[1] = -1.2
        b[2] = -0.1
        expected_BT = np.zeros(g.number_of_edges())
        expected_BT[0] = -1
        expected_BT[1] = 1.2
        expected_BT[2] = 0.1
        actual_BT = sherman_flow.compute_BT(b)
        npt.assert_array_equal(expected_BT, actual_BT)

    def test_compute_R(self):
        g = graph_util.complete_graph(5)
        congestion_approximator = ConductanceCongestionApprox(g)
        sherman_flow = sherman.ShermanFlow(g, congestion_approximator)

        x = np.ones(g.number_of_edges())
        actual_Rx = sherman_flow.compute_R(x)
        actual_RTx = sherman_flow.compute_RT(x)
        
        self.assertEqual(actual_Rx.shape, (g.number_of_edges(),))
        self.assertEqual(actual_RTx.shape, (g.number_of_nodes(),))

    def test_max_flow_conductance_cong_approx(self):
        epsilon = 0.1
        n = 10
        for p in [0.7, 0.8, 0.9, 1.0]:
            for _ in range(100):
                g = graph_util.diluted_complete_graph(n, p)
                if not g.has_edge(0, 1):
                    g.add_edge(0, 1, capacity=1)
                cong_approx = ConductanceCongestionApprox(g)
                sherman_flow = sherman.ShermanFlow(g, cong_approx)
                _, flow_value = sherman_flow.max_st_flow(0, 1, epsilon)
                actual_flow_value, _ = nx.maximum_flow(g.to_undirected(), 0, 1)
                self.assertGreaterEqual(flow_value, (1.0 - epsilon) * actual_flow_value)
                self.assertLessEqual(flow_value, (1.0 + epsilon) * actual_flow_value)

    def test_max_flow_mst_cong_approx(self):
        epsilon = 0.1
        for _ in range(20):
            g = graph_util.diluted_complete_graph(10, 0.9)
            if not g.has_edge(0, 1):
                g.add_edge(0, 1, capacity=1)
            cong_approx = MstCongestionApprox(g.to_undirected())
            sherman_flow = sherman.ShermanFlow(g, cong_approx)
            _, flow_value = sherman_flow.max_st_flow(0, 1, epsilon)
            actual_flow_value, _ = nx.maximum_flow(g.to_undirected(), 0, 1)
            self.assertGreaterEqual(flow_value, (1.0 - epsilon) * actual_flow_value)
            self.assertLessEqual(flow_value, (1.0 + epsilon) * actual_flow_value)

if __name__ == '__main__':
    unittest.main()
