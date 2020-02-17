import numpy as np
from scipy import random, linalg

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
    return dist_mat

def random_data(N, kernel_type="rbf", sigma=0.1):
    """
    input: 
    N - number of data points to generate
    kernel - which kernel to use, for now only rbf is implemented

    output:
    X - data
    K - kernel matrix
    """
    # choose a random dimension for the data
    dimension = 100
    X = random.rand(N, dimension)

    # get a kernel matrix
    K = compute_kernel(X, kernel_type, sigma)

    return X, K

def compute_kernel(X, kernel_type="rbf", sigma=0.1):
    """
    input:
    X - data
    kernel - kernel type

    output:
    K - kernel matrix
    """
    K = np.zeros((X.shape[0], X.shape[0]))
    if kernel_type == "rbf":
        # find the distance matrix
        dist_mat = compute_squared_distance_no_loops(X, X)
        # divide by 2*sigma^2
        K = dist_mat/(-2*np.power(sigma, 2))
        K = np.exp(K)

    return K
