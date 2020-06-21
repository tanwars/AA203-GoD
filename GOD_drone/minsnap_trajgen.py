import numpy as np
from scipy.linalg import block_diag
import cvxpy as cvx

import plot_utils
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

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


def getA(t):
    '''
    assumes snap
    '''
    A = np.array([[1, t, t**2, t**3, t**4, t**5, t**6, t**7],
                [0, 1, 2*t, 3*(t**2), 4*(t**3), 5*(t**4), 6*(t**5), 7*(t**6)],
                [0, 0, 2, 6*t, 4*3*(t**2), 5*4*(t**3), 6*5*(t**4), 7*6*(t**5)],
                [0, 0, 0, 6, 4*3*2*(t), 5*4*3*(t**2), 6*5*4*(t**3), 7*6*5*(t**4)]])
    return A

def minSnapTG_without_opt(N, n, T, W, W_vel = None, idx_vel = None, 
                                W_acc = None, idx_acc = None,
                                W_jerk = None, idx_jerk = None,):
    '''
    only works for snap
    '''

    M = len(T)
    
    # For each segment, compute Q and A
    Q = np.zeros((M, N, N), dtype=np.float64)
    AM = np.zeros((M*N, M*N), dtype=np.float64)
    for m in range(M):
        # Minimizing derivative Hessian matrix Q
        for i in range(n, N+1):
            for j in range(n, N+1):
                k = np.arange(0, n, dtype=np.float64)
                Q[m,i-1,j-1] = 2 * np.prod((i-k)*(j-k)) * T[m]**(i+j+1-2*n) / (i+j+1-2*n)
        
        t_end = T[m]

        if m == 0:
            AM[:n,:N] = getA(0)
            AM[n:2*n, :N] = getA(t_end)
            prev_t_end = t_end
        else:
            AM[m*N:m*N+n, (m-1)*N:m*N] = getA(prev_t_end)
            AM[m*N:m*N+n, m*N: (m+1)*N] = -getA(0)
            AM[m*N+n:m*N+2*n, m*N: (m+1)*N] = getA(t_end)
        
        prev_t_end = t_end

    # Assemble block diagonal matrices Q1...M, A1...M
    QM = block_diag(*Q)

    # find idx of fixed and non-fixed derivatives
    fixed_deriv = []
    unfixed_deriv = []

    # position
    for idx in range(M*N):
        if np.mod(idx, n) == 0:
            fixed_deriv.append(idx)
        elif idx>N and np.mod(idx, N)<n:
            fixed_deriv.append(idx)
        else:
            unfixed_deriv.append(idx)

    # construct the C matrix
    C = np.zeros((M*N, M*N))
    nf = len(fixed_deriv)
    nuf = len(unfixed_deriv)

    for i in range(nf):
        C[i, fixed_deriv[i]] = 1
    for i in range(nuf):
        C[i+nf, unfixed_deriv[i]] = 1

    # construct d
    d = np.zeros((M*N,))
    W_count = 0
    for idx in range(M*N):
        if np.mod(idx, n) == 0:
            if idx<N:
                if idx ==0:
                    d[idx] = W[W_count]
                    W_count += 1
                else:
                    d[idx] = W[W_count]
                    W_count += 1
            elif idx>N and np.mod(idx, N)==n:
                d[idx] = W[W_count]
                W_count += 1

    # Minimization
    H = C @ np.linalg.inv(AM).T @ QM @ np.linalg.inv(AM) @ C.T

    Rff = H[:nf, :nf]
    Rfp = H[:nf, nf:]
    Rpf = H[nf:, :nf]
    Rpp = H[nf:, nf:]

    d_star_p = - np.linalg.inv(Rpp) @ Rfp.T @ C[:nf,:] @ d
    d_star = np.hstack((C[:nf,:] @ d, d_star_p))

    d = C.T @ d_star

    p = np.linalg.solve(AM, d)

    P = np.zeros((N, M))
    for i in range(M):
        P[:,i] = np.flip(p[i*N:(i+1)*N])
    return P

# N = 8
# n = 4
# T = np.array([1, 1, 1], dtype=np.float64)
# W = np.array([0, 1, 2, 5], dtype=np.float64)

# # P1 = minSnapTG(N, n, T, W)
# # print(P1)
# # plot_utils.plot_single_traj(P1, T, 0.1)

# P2 = minSnapTG_without_opt(N, n, T, W)
# print(P2)
# plot_utils.plot_single_traj(P2, T, 0.1)


# plt.show()