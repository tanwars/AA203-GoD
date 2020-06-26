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

def find_deriv_idx(idx_list, M, N, n, this_num):
    '''
    this num: 0,1,2,3
        0 - position
        1 - vel
        2 - acc
        3 - jerk
    '''
    
    if idx_list is None:
        return []
    
    this_list = []

    for m in idx_list: # m is waypoint number
        if m == 0:
            this_list.append(this_num)
        else:
            this_list.append(this_num + (m-1)*N + n)

    return this_list

def find_zero_idx(M, N, n):
    
    this_list = []
    
    for m in range(M):
        if m == 0:
            continue

        for idx in range(n):
            this_list.append(m*N + idx)

    return this_list

def find_unfixed_deriv(fixed_deriv, M, N):

    arr1 = np.array(fixed_deriv)
    arr2 = np.arange(N * M)

    this_list = list(np.setdiff1d(arr2, arr1, assume_unique=True))
    return this_list

def minSnapTG_without_opt_new(N, n, T, W = None, idx_pos = None, 
                                        W_vel = None, idx_vel = None, 
                                        W_acc = None, idx_acc = None,
                                        W_jer = None, idx_jer = None):
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

    pos_fixed_idx = find_deriv_idx(idx_pos, M, N, n, 0)
    vel_fixed_idx = find_deriv_idx(idx_vel, M, N, n, 1)
    acc_fixed_idx = find_deriv_idx(idx_acc, M, N, n, 2)
    jer_fixed_idx = find_deriv_idx(idx_jer, M, N, n, 3)
    fixed_deriv.append(pos_fixed_idx)
    fixed_deriv.append(vel_fixed_idx)
    fixed_deriv.append(acc_fixed_idx)
    fixed_deriv.append(jer_fixed_idx)

    fixed_deriv.append(find_zero_idx(M, N, n))

    fixed_deriv = [item for sublist in fixed_deriv for item in sublist]
    
    unfixed_deriv = find_unfixed_deriv(fixed_deriv, M, N)

    # print(fixed_deriv)
    # print(unfixed_deriv)

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

    d[pos_fixed_idx] = W
    d[vel_fixed_idx] = W_vel
    d[acc_fixed_idx] = W_acc
    d[jer_fixed_idx] = W_jer

    # print(d)

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

    print('Cost NonCVX: %f'% (p.T @ QM @ p))

    P = np.zeros((N, M))
    for i in range(M):
        P[:,i] = np.flip(p[i*N:(i+1)*N])
    return P

# N = 8
# n = 4
# T = np.array([5, 5, 5], dtype=np.float64)
# W = np.array([0, 1, 2, 5], dtype=np.float64)
# V = np.array([], dtype=np.float64)

# vel_idx = np.array([])
# pos_idx = np.array([0,1,2,3])

# # P1 = minSnapTG(N, n, T, W)
# # print(P1)
# # plot_utils.plot_single_traj(P1, T, 0.1)

# # P2 = minSnapTG_without_opt(N, n, T, W)
# # print(P2)
# # plot_utils.plot_single_traj(P2, T, 0.1)

# P3 = minSnapTG_without_opt_new(N, n, T, W = W, idx_pos= pos_idx, W_vel = V, 
#                                                             idx_vel = vel_idx)
# # print(P2)
# plot_utils.plot_single_traj(P3, T, 0.1)

# plt.show()