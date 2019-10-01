import numpy as np
from scipy import random, linalg

def sampler_indices(X, s, pi):
    """
    for each i in X, choose X[i] with probability pi[i] independently
    """
    chosen_indices = []
    for i in range(len(X)):
        rand_val = np.random.rand()
        if pi[i] == 1:
            chosen_indices = [i] + chosen_indices
        else:
            if pi[i] >= rand_val:
                chosen_indices.append(i)
            else:
                pass
    chosen_indices = chosen_indices[0:s]
    return chosen_indices

def rls_sampling(X, K, _lambda, s, delta):
    """
    input:
    X - data
    K - kernel
    _lambda - ridge parameter
    s - number of samples used to reconstruct Nystrom
    delta - failure probability

    output:
    K_tilde - kernel approximation
    """
    if s > X.shape[0]:
        print("you can't approximate further")
        sys.exit(1)
    # compute over approximation li for _lambda ridge leverage scores
    li = np.diagonal(K*np.linalg.inv(K+_lambda*np.eye(K.shape[0])))
    # set oversampling parameter
    q = 0.1
    # compute oversampled li
    li = q*li
    # set pi = min{1, li*16*log(\sum li/delta)}
    sum_li = np.sum(li)
    pi = li*16*np.log(sum_li/delta)
    pi = np.minimum(1, pi)
    # construct S by sampling x1,..,xn each with prob pi
    # choose indices first
    chosen_indices = sampler_indices(X, s, pi)#np.random.choice(list(range(X.shape[0])), s, p=pi, replace=False)
    # construct S
    S = np.zeros((X.shape[0], len(chosen_indices)))
    # fill in S
    for i in range(len(chosen_indices)):
        S[chosen_indices[i], i] = 1
    # KS, STKS
    KS = K.dot(S)
    SKS = S.T.dot(K.dot(S))
    return KS, SKS

def recursive_rls(X, K_func, _lambda, s, delta):
    """
    input:
    X - data
    K_func - kernel function
    _lambda - ridge parameter
    s - number of samples used to reconstruct Nystrom
    delta - failure probability

    output:
    K_tilde - kernel approximation
    """
    m = X.shape[0]
    # step 1: check if m is large enough
    if m <= 192*log(1/delta):
        return np.eye(m)
    # choose a random subset of X
    indices = list(range(m))
    indices = np.random.permutation(indices)
    number_of_elements = np.random.randint(low=1, high=m)
    chosen_indices = indices[0:number_of_elements]
    X_sampled = X[chosen_indices, :]
    # compute sampling matrix
    S_bar = range(len(X_sampled))
    # recurse
    S_tilde = recursive_rls(X_sampled, K_func, _lambda, s, delta)
    # compute S_hat
    S_hat = S_bar*S_tilde
    # compute li
    K = K_func(X)
    SKS = S_hat.T.dot(K.dot(S_hat))
    SK = S_hat.T.dot(K)
    KS = K.dot(S)
    li = np.diagonal((3/(2*_lambda))*(K - (KS.dot(np.linalg.inv(SKS-_lambda*np.eye(SKS.shape[0])))).dot(SK)))
    # compute pi
    sum_li = np.sum(li)
    pi = np.minimum(1, li*16*np.log(sum_li/delta))
    # set weight sampling matrix to be empty
    S = np.zeros((m, s))
    # choose indices
    chosen_indices = sampler_indices(S_bar, s, pi)#np.random.choice(list(range(S_bar.shape[0])), s, p=pi, replace=False)
    # compute S
    for i in range(len(S)):
            S[i, :] = S_bar[chosen_indices[i], :] / pi[chosen_indices[i]]
    
    return S
