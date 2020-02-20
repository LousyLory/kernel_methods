import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from utils import compute_kernel

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
    # Ksub = Let K(i,j) = e^-(-1/(2*gamma)*||X(i,:)-X(j,:)||^2). Then Ksub =
    # K(rowInd,colInd). Or if colInd = [] then Ksub = diag(K)(rowInd).

    if _type_ == "rbf":
        if len(colInd) <= 0:
            Ksub = np.ones((len(rowInd)))
        else:
            Ksub = compute_kernel(Data, rowInd, colInd, kernel_type=_type_, sigma=gamma)
    else:
        pass

    return Ksub

def online_RLS(x, KS, SKS, _iter):
    # check of the kernel and selection matrices are empty to begin with
    if _iter == 0:
        S = []
        K = []
    
    # compute the number of fetaures of x
    d = x.shape[1]
    # set the random variables
    eps = 0.5
    c = 8.0*np.log(d) / (np.log(2)*eps**2)





