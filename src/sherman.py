from __future__ import division
import numpy as np
import numpy.linalg as la
import math
import graph_util
from soft_max import soft_max, grad_soft_max
from conductance_congestion_approx import ConductanceCongestionApprox

class ShermanFlow:
    def __init__(self, g, cong_approx):
        self.graph = g
        self.cong_approx = cong_approx
        self.edge_capacities = [1.0 * c for c in graph_util.get_edge_capacities(g)]
        self.edge_capacities_inv = [1.0 / c for c in graph_util.get_edge_capacities(g)]

    def compute_R(self, x):
        return np.array(self.cong_approx.compute_dot(x))

    def compute_RT(self, x):
        return np.array(self.cong_approx.compute_transpose_dot(x))

    def compute_C(self, x):
        return np.multiply(x, self.edge_capacities)

    def compute_Cinv(self, x):
        return np.multiply(x, self.edge_capacities_inv)

    def compute_B(self, x):
        excess = np.zeros(self.graph.number_of_nodes())
        edge_i = 0
        for edge_from, edge_to in graph_util.edge_iter(self.graph):
            excess[edge_from] -= x[edge_i]
            excess[edge_to] += x[edge_i]
            edge_i += 1
        return excess

    def compute_BT(self, x):
        potentials = np.zeros(self.graph.number_of_edges())
        edge_i = 0
        for edge_from, edge_to in graph_util.edge_iter(self.graph):
            potentials[edge_i] = -x[edge_from] + x[edge_to]
            edge_i += 1
        return potentials

    def phi(self, f, b):
        alpha = self.cong_approx.alpha()
        resid = b - self.compute_B(f)
        return soft_max(self.compute_Cinv(f)) + soft_max(
            2 * alpha * self.compute_R(resid))

    def grad_phi(self, f, b):
        x1 = self.compute_Cinv(f)
        p1 = grad_soft_max(x1)

        resid = b - self.compute_B(f)
        alpha = self.cong_approx.alpha()
        x2 = 2 * alpha * self.compute_R(resid)
        p2 = grad_soft_max(x2)

        return self.compute_Cinv(p1) - 2 * alpha * (
            self.compute_BT(self.compute_RT(p2)))

    def almost_route(self, demands, epsilon):
        n = self.graph.number_of_nodes()
        m = self.graph.number_of_edges()

        # TODO: fiddle around with these constants. They come from
        #   (loose) bounds in the correctness proof, and have a
        #   significant effect on performance.
        k1 = 7 / 2 / epsilon
        k2 = 2 / 7

        scaling = 1
        f = np.zeros(m)
        y = np.array(f)
        b = np.array(demands)
        norm_Rb = la.norm(self.compute_R(b), np.inf)
        alpha = self.cong_approx.alpha()
        scaling *= abs(k1 * math.log(n) / (2 * alpha * norm_Rb))
        b = b * scaling
        iters = 1

        while True:
            while self.phi(f, b) < k1 * math.log(n):
                f = (k1 + 1) / k1 * f
                y = (k1 + 1) / k1 * y
                b = (k1 + 1) / k1 * b
                scaling *= (k1 + 1) / k1

            grad_phi_y = self.grad_phi(y, b)
            delta = la.norm(self.compute_C(grad_phi_y), 1)
            if delta >= k2 * epsilon:
                f_prev = np.array(f)
                f = y - delta / (1 + 4 * alpha**2) * self.compute_C(
                    np.sign(grad_phi_y))
                y = f + (iters - 1) / (iters + 2) * (f - f_prev)
                iters += 1
            else:
                return f / scaling

    def min_congestion_flow(self, demands, epsilon):
        n = self.graph.number_of_nodes()
        m = self.graph.number_of_edges()
        f_total = np.zeros(m)
        for i in range(int(math.log(2 * m))):
            f = self.almost_route(demands, epsilon)
            demands = demands - self.compute_B(f)
            epsilon = 0.5
            f_total += f
        return f_total

    def max_flow(self, demands, epsilon):
        flow = self.min_congestion_flow(demands, epsilon)
        max_edge_congestion = la.norm(self.compute_Cinv(flow), np.inf)
        max_flow = flow / max_edge_congestion
        max_flow_value = 0
        sink_nodes = np.maximum(np.sign(demands), np.zeros(len(demands)))
        max_flow_value = np.dot(self.compute_B(max_flow), sink_nodes)
        return max_flow, max_flow_value

    def max_st_flow(self, source_i, sink_i, epsilon):
        demands = np.zeros(self.graph.number_of_nodes())
        demands[source_i] = -1
        demands[sink_i] = 1
        return self.max_flow(demands, epsilon)
