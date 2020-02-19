# Copyright (c) 2019 Archan Ray
#
#------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import random as rnd

def compute_squared_distance_no_loops(X, Y):
    """
        input:
        X - train_data
        Y - test_data

        output:
        dist_mat - computed distance values
    """
    num_test = Y.shape[0]
    num_train = X.shape[0]
    dist_mat = np.zeros((num_test, num_train))

    # compute distance
    sum1 = np.sum(np.power(Y,2), axis=1)
    sum2 = np.sum(np.power(X,2), axis=1)
    sum3 = 2*np.dot(Y, X.T)
    dists = sum1.reshape(-1,1) + sum2
    dist_mat = dists - sum3
    return dist_mat.T

def recursiveNystrom(X, s, kernelFunc):
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

    [n, d] = np.shape(X)

    # start of algoithm
    oversamp = np.log(s)
    k = np.ceil(s/4*oversamp)
    nLevels = int(np.ceil(np.log(n/s)/np.log(2)))
    # randomly permuting for successful uniform samples
    perm = np.random.permutation(n)

    # set up sizes for recursive levels
    lSize = np.zeros((nLevels+1))
    lSize[0] = n
    for i in range(1, nLevels+1):
        lSize[i] = int(np.ceil(lSize[i-1]/2))
    
    # rInd: indices of points selected at previous level
    # of recursion. At base level its a uniform sample of 
    # about s points
    samp = list(range(int(lSize[-1])))
    rInd = perm[samp]
    weights = np.ones((len(rInd)))

    # we need dagonal fo the whole kernel matrix
    kDiag = kernelFunc(X, list(range(n)), [])

    # main recursion
    for l in range(nLevels, 0, -1):
        # indices of current uniform sample
        rIndCurr = np.random.permutation(lSize[l])

    C = 0
    W = 0
    
    return C, W

def kernelFunction(Data, rowInd, colInd, gamma=0.1, _type_="rbf"):
    # a kernel generator: outputs the submatrix of the associated kernel 
    # with variance parameter.
    # input:
    #
    # Data: Data matrix with n points and d features
    # rowInd, colInd: List of indices between [1,n] for each row & col
    # gamma: kernel variance parameter
    #
    # output:
    # 
    # Ksub = Let K(i,j) = e^-(gamma*||X(i,:)-X(j,:)||^2). Then Ksub = 
    # K(rowInd,colInd). Or if colInd = [] then Ksub = diag(K)(rowInd).

    if _type_ == "rbf":
        if not colInd:
            Ksub = np.ones((len(rowInd)))
        else:
            Ksub = np.exp(-1*gamma*compute_squared_distance_no_loops(Data[rowInd, :], Data[colInd, :]))

    return None

def main():
    X = np.random.random((1000, 500))
    s = 40
    kernelFunc = kernelFunction
    recursiveNystrom(X, s, kernelFunc)

    return None

if __name__ == "__main__":
    main()
