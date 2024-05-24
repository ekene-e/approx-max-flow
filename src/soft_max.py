from __future__ import division
import numpy as np
import math

def soft_max(x):
    summation = np.sum(np.exp(x) + np.exp(-x))
    return math.log(summation)

def grad_soft_max(x):
    ex = np.exp(x)
    emx = np.exp(-x)
    summation = np.sum(ex + emx)
    return (ex - emx) / summation
