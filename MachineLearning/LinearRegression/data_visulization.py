import matplotlib.pyplot as plt
import numpy as np

experiences = np.array([0,1,2,3,4,5,6,7,8,9,10])
salaries = np.array([103100, 104900, 106800, 108700, 110400, 112300, 114200, 116100, 117800, 119700, 121600])

X = experiences.reshape(-1,1)
Y = salaries

plt.scatter(X, Y,  color='black')

plt.show()