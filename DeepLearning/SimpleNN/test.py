import numpy as np
from scipy.special import expit

a = [1,2]
b = [3,4]

print(np.dot(a,b))

a = np.array([1,2,3])
b = np.array([[1,4,7],[2,5,8],[3,6,9]])
a_T = np.array([[1],[2],[3]])

c = np.dot(a,b)
d = np.dot(b,a)
print(c)
print(d)

e = np.dot(b,a_T)
print(e)

f = np.add(c, a)
print(f)


z = np.array([ 0, 0, 0])
g = expit(z)

print(g)