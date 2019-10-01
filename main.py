import numpy as np
from scipy import linalg, random
from nystrom import *
from utils import *

"""
generate random PSD matrix
"""
"""
# number of samples
N = 10
# generate a random matrix of shape NxN
random_matrix = random.rand(N, N)
# convert to PSD
K = np.dot(random_matrix, random_matrix.transpose())
"""

# generate data
N = 30 # number of smaples
X, K = random_data(N)

# check for PSD (well the check for much stricter PD)
print(np.all(np.linalg.eigvals(K) > 0))

# recover RLS nystrom sampling
_lambda = 0.01
delta = random.uniform(0.0, 1.0/8)
s = 10
K_tilde = rls_sampling(X, K, _lambda, s, delta)
