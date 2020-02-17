import numpy as np
from scipy import linalg, random
from nystrom import *
from utils import *
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# generate data
N = 30 # number of smaples
sigma1 = 1.01
sigma2 = 1.02
for i in tqdm(range(5000)):
	X1, K1 = random_data(N, sigma=sigma1)
	K2 = compute_kernel(X1, sigma=sigma2)
	K3 = K2-K1

	set_lambda = 3.0

	u1, s1, v1 = np.linalg.svd(K1)
	u2, s2, v2 = np.linalg.svd(K2)
	u3, s3, v3 = np.linalg.svd(K3)

	if np.allclose(K1, K2):
	    print("same matrices returned")

	"""
	checking results
	"""

	"""
	print("Sum of first eigenvalues:", np.sum(s1))
	print("First trace:", np.trace(K1))
	print("Sum of squares of eigenvalues for first kernel:", np.sum(s1**2))
	print("Determinant of first kernel", np.linalg.det(K1))
	print("max eigval of K1", s1[0])

	print("Sum of second eigenvalues:", np.sum(s2))
	print("Second trace:", np.trace(K2))
	print("Sum of squares of eigenvalues for second kernel:", np.sum(s2**2))
	print("Determinant of second kernel", np.linalg.det(K2))
	print("max eigval of K2", s2[0])

	print("max eigval of K3", s3[0]) 
	print("max eigval of K1+K3 > K2:", s1[0]+s3[0] > s2[0])
	"""
	#print("det(K2-K1) <= det(K3):", np.linalg.det(K2-K1) <= np.linalg.det(K3))
	#print("det(K2-K1) >= det(K2)-det(K1):", np.linalg.det(K2-K1) >= np.linalg.det(K2)-np.linalg.det(K1))
	#print("det(K2) >= det(K1)+det(K3):", np.linalg.det(K2) >= np.linalg.det(K1)+np.linalg.det(K3))
	if np.linalg.det(K2) >= np.linalg.det(K1)+np.linalg.det(K3):
		print("det(K2) >= det(K1)+det(K3) is not true")
	"""
	print("effective stat dimension of K1:", np.sum(s1 / (s1+set_lambda)))
	print("effective stat dimension of K2:", np.sum(s2 / (s2+set_lambda)))

	sns.set(style="darkgrid")
	plt.title("comparison of eigenvalues")
	plt.plot(s1, label="sigma="+str(sigma1))
	plt.plot(s2, label="sigma="+str(sigma2))
	plt.xlabel("eigenvalue indices")
	plt.ylabel("eigenvalues")
	plt.legend(loc="upper right")
	plt.savefig("comparison_of_eigenvalues"+".pdf")
	"""
