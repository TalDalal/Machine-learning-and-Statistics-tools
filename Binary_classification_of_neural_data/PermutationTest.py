import numpy as np
'''
Permutation Test - shuffle two vector and constract the theoretical distribution according to central limit theorm,
check for the probability of the empirical value.
'''
# randomize two vectors from exponential distribution.
n = 25
x1 = np.random.exponential(2, (n, 1))
x2 = np.random.exponential(2, (n, 1))

# we combine x1 and x2 to X, and S will be the suffled vector.
iteration_num = 1000
X = np.append(x1, x2)
S = np.zeros(iteration_num)

# our statistic parameter that we check is difference between means.
S[0] = np.mean(x1) - np.mean(x2)

for iteration in np.arange(1,len(X)):
    rand = np.random.permutation(len(X))
    p1 = rand[0:len(x1)]
    p2 = rand[len(x1)+1:len(X)]

    S[iteration] = np.mean(X[p1]) - np.mean(X[p2])


    

