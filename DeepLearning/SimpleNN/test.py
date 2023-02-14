import numpy as np
from scipy.special import expit


M = np.array([[3,3,3],[2,2,2]])

va = np.array([1,1])
vb = np.array([1,1,1])

ra = np.dot(va, M)
print(ra)
rb = np.dot(M, vb)
print(rb)