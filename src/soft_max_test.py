from __future__ import division
import numpy as np
import numpy.linalg as la
import math
from soft_max import soft_max, grad_soft_max
import unittest

class SoftMaxTest(unittest.TestCase):
    def test_grad_norm_inequalities(self):
        for n in range(2, 100):
            for i in range(1000):
                x = np.random.normal(scale=n, size=(n))
                lmax_x = soft_max(x)
                grad_x = grad_soft_max(x)

                self.assertLessEqual(la.norm(grad_x, 1), 1 + 1e-8)
                self.assertGreaterEqual(grad_x.dot(x), lmax_x - math.log(2*n))

                y = np.random.normal(scale=n, size=(n))
                lmax_y = soft_max(y)
                grad_y = grad_soft_max(y)
                self.assertLessEqual(la.norm(grad_x - grad_y, 1), la.norm(x - y, np.inf))

if __name__ == '__main__':
    unittest.main()
