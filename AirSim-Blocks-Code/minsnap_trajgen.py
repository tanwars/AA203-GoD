'''
Adam's code for minsnap trajectory generation
'''

import numpy as np
from scipy.linalg import block_diag
import cvxpy as cvx


# N - number of coefficients (degree + 1)
# n - order derivative to minimize 
# T - segment time lengths 
# W - waypoints
def minSnapTG(N, n, T, W):
    
    M = len(T)
    
    # For each segment, compute Q and A
    Q = np.zeros((M, N, N), dtype=np.float64); A = np.zeros((M, N, N), dtype=np.float64)
    for m in range(M):
        # Minimizing derivative Hessian matrix Q
        for i in range(n, N+1):
            for j in range(n, N+1):
                k = np.arange(0, n, dtype=np.float64)
                Q[m,i-1,j-1] = np.prod((i-k)*(j-k)) * T[m]**(i+j+1-2*n) / (i+j+1-2*n)
        
        # Derivative constraint matrix A
        for i in range(n): # order of derivative
            A[m,i,i] = np.math.factorial(i)
            for j in range(N): # order of polynomial term
                if j >= i:
                    A[m,i+n,j] = (np.math.factorial(j) / np.math.factorial(j-i)) * T[m]**(j-i)

    # Assemble block diagonal matrices Q1...M, A1...M
    QM = block_diag(*Q)
    AM = block_diag(*A)

    # Minimization
    H = np.linalg.inv(AM).T @ QM @ np.linalg.inv(AM)

    d = cvx.Variable(2*n*M) 

    # Initial state constraints - derivatives fixed to 0
    constraints = [d[0] == W[0], d[1:n] == 0]

    # Waypoint and continuity constraints
    for i in range(1,M):
        j = 2*i - 1
        constraints += [d[j*n] == W[i]] # waypoint
        constraints += [d[j*n:(j+1)*n] == d[(j+1)*n:(j+2)*n].copy()] # continuity

    # Final waypoint constraint
    constraints += [d[(2*M-1)*n] == W[M]] 
    
    objective = cvx.Minimize(cvx.quad_form(d, cvx.Parameter(shape=H.shape, value=H, PSD=True)))

    prob = cvx.Problem(objective, constraints)
    prob.solve()

    p = np.linalg.solve(AM, d.value)

    P = np.zeros((N, M))
    for i in range(M):
        P[:,i] = np.flip(p[i*N:(i+1)*N])
    return P

# N = 8
# n = 4
# T = np.array([1, 3], dtype=np.float64)
# W = np.array([0, 3, 8], dtype=np.float64)

# P = minSnapTG(N, n, T, W)
# print(P)