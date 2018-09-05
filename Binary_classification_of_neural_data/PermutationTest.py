import numpy as np
n = 25
iteration_num = 1000
x1 = np.random.exponential(2, (n, 1))
x2 = np.random.exponential(2, (n, 1))

X = np.append(x1, x2)
S = np.zeros(2000)

for iteration in range(iteration_num):
    rand = np.random.permutation(len(X))
    

