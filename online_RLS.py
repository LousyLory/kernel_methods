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

def online_RLS(x, X, KS, SKS, _iter, sample_indices, weight):
    # check of the kernel and selection matrices are empty to begin with
    if _iter == 0:
        SKS = np.array([])
        KS = np.array([])
    
    # compute the number of fetaures of x
    d = x.shape[1]

    # set the parameter variables
    _lambda = 0.1
    eps = 0.5
    #c = 8.0*np.log(d) / (eps**2)
    c = 0.2

    # first pre sample
    sample_indices_new = sample_indices+[len(X)]
    weight_new = weight+[1]

    # add new data steam to previous KS and SKS
    X_new = np.append(X, x, axis=0)
    KS_new_col = kernelFunction(X_new, list(range(len(X_new))), [-1])
    KS_new = np.append(KS, KS_new_col, axis=1)
    # using square root of the weights
    KS_new = KS_new*np.sqrt(np.array(weight_new))
    SKS_new = KS_new[sample_indices_new, :]*np.array(weight_new)
    SKS_lambdaI = SKS_new+_lambda*np.eye_like(SKS_new)
    inv_SKS_lambdaI = inv(SKS_lambdaI)

    # now compute the product finally (I think this needs optimization for not storing the n x n matrix, 
    # ask cameron about this)
    intermediate_prod = np.matmul(KS_new, inv_SKS_lambdaI)
    prod = np.dot(intermediate_prod[-1, :], KS_new[-1, :])

    # rejection hypothesis
    levs = 3.0/(2.0*_lambda) * (1.0 - prod)
    pi = np.min(c*levs, 1.0)

    si_choice = [0, 1]
    si_final = np.random.choice(si_choice, 1, p=[1-pi, pi])
    si_final = si_final[0] / np.sqrt(pi)

    if si_final > 0:
        weight = weight+[si_final]
        sample_indices = sample_indices + [len(X)]
        KS = KS_new
        KS = KS[:,-1] / si_final
        SKS = KS[sample_indices, :] * np.array(weight)

        return KS, SKS, sample_indices, weight, X
    else:
        return KS, SKS, sample_indices, weight, X


def update(x, X, KS, SKS, t, T, sample_indices, weight):
    # the benevolent toddler
    sample_indices_new = sample_indices+[len(X)]
    weight_new = weight+[1]

    X_new = np.append(X, x, axis =0)
    KS_new_col = kernelFunction(X_new, list(range(len(X_new))), [-1])
    KS_new = np.append(KS, KS_new_col, axis=1)

    KS_new = KS_new*np.sqrt(np.array(weight_new))
    SKS_new = KS_new[sample_indices_new, :]*np.array(weight_new)

    T_new = T+[t]

    return KS_new, SKS_new, sample_indices_new, weight_new, T_new, X_new

def expire(KS, SKS, sample_indices, weight, T, X, W, t):
    # this is purger
    for i in range(len(T)):
        if T[i] <= t-W+1:
            ID = sample_indices[i]
            sample_indices.remove(ID)
            sample_indices[i:] = sample_indices[i:]-1

            weight.pop(i)
            T.pop(i)

            X = np.delete(X, (ID), axis=0)
            KS = np.delete(KS, (i), axis=1)
            KS = np.delete(KS, (ID), axis=0)
            SKS = np.delete(SKS, (i), axis=0)
            SKS = np.delete(SKS, (i), axis=1)

        else:
            pass

    return KS, SKS, sample_indices, weight, T, X

def downsample(KS, SKS, sample_indices, weight, X):
    # MVP for the windowed algorithm
    eps = 0.5
    # c = ((3+eps)/(3*(eps**2))) * 2 * np.log(X.shape[1])
    c = 0.2

    KS_new = np.array([])
    SKS_new = np.array([])
    X_new = np.array([])
    weight_new = []
    sample_indices_new = []

    for i in range(len(sample_indices)-1, -1, -1):
        weight_new = weiht_new+[1]
        sample_indices_new = sample_indices_new+[i]
        xi = X[sample_indices[i]]
        X_new = np.append(X_new, xi)
        KS_new_col = kernelFunction(X_new, list(range(len(X_new))), [-1])
        KS_new = np.append(KS_new, KS_new_col, axis=1)

    return KS, SKS, sample_indices, weight


def metaspectral(x, X, KS, SKS, sample_indices, weight, t, T):
    # wrapper for the windowed algorithm
    W = 50
    if t == 1:
        KS = np.array([])
        SKS = np.array([])
    else:
        pass

    KS, SKS, sample_indices, weight, T, X = update(x, X, KS, SKS, t, T, sample_indices, weight)
    KS, SKS, sample_indices, weight = downsample(KS, SKS, sample_indices, weight, X)
    KS, SKS, sample_indices, weight, T, X = expire(KS, SKS, sample_indices, weight, T, X, W, t)

    return KS, SKS, sample_indices, weight, T, X