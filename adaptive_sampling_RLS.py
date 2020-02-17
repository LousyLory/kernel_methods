# Copyright (c) 2019 Archan Ray
#
#------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import random as rnd

def recursiveNystrom(X, s, kernelFunction):
    # implements Algorithm 3 of paper https://arxiv.org/abs/1605.07583
    #
    # input::
    #
    # X: matrix with n rows each with d features.
    #
    # s: sample size for Nystrom approximation. Generally s << n.
    #
    # kernelFunction: a function that can compute arbitary sub-
    # matrices of a PSD kernel. 
    #
    # output::
    #
    # C: a subset of s columns from X's n x n kernels.
    #
    # W: an s x s PSD such that C*W*C' approximates K.
    
    return C, W
