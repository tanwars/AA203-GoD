import numpy as np
from scipy.linalg import block_diag
import torch
import matplotlib.pyplot as plt
torch.autograd.set_detect_anomaly(True)

def block_diag(m):
    """
    Make a block diagonal matrix along dim=-3
    EXAMPLE:
    block_diag(torch.ones(4,3,2))
    should give a 12 x 8 matrix with blocks of 3 x 2 ones.
    Prepend batch dimensions if needed.
    You can also give a list of matrices.
    :type m: torch.Tensor, list
    :rtype: torch.Tensor
    """
    if type(m) is list:
        m = torch.cat([m1.unsqueeze(-3) for m1 in m], -3)

    d = m.dim()
    n = m.shape[-3]
    siz0 = m.shape[:-3]
    siz1 = m.shape[-2:]
    m2 = m.unsqueeze(-2)
    eye = attach_dim(torch.eye(n).unsqueeze(-2), d - 3, 1)
    return (m2 * eye).reshape(
        siz0 + torch.Size(torch.tensor(siz1) * n)
    )

def attach_dim(v, n_dim_to_prepend=0, n_dim_to_append=0):
    return v.reshape(
        torch.Size([1] * n_dim_to_prepend)
        + v.shape
        + torch.Size([1] * n_dim_to_append))


# n - order derivative to minimize 
# T - segment time lengths 
# W - waypoints
def minSnapCostMat(n, T, W):

    M = T.shape[0]  # number of segments
    N = 2*n     # number of polynomial coefficients

    # For each segment, compute Q and A
    Q = torch.zeros((M, N, N), dtype=torch.float64); A = torch.zeros((M, N, N), dtype=torch.float64)
    for m in range(M):
        # Minimizing derivative Hessian matrix Q
        for i in range(n, N+1):
            for j in range(n, N+1):
                k = np.arange(0, n, dtype=np.float64)
                Q[m,i-1,j-1] = np.prod((i-k)*(j-k)) * T[m].clone()**(i+j+1-2*n) / (i+j+1-2*n)
                #Q[m,i-1,j-1] = np.prod((i-k)*(j-k)) * torch.pow(T[m], (i+j+1-2*n)) / (i+j+1-2*n)
        
        # Derivative constraint matrix A
        for i in range(n): # order of derivative
            A[m,i,i] = np.math.factorial(i)
            for j in range(N): # order of polynomial term
                if j >= i:
                    A[m,i+n,j] = (np.math.factorial(j) / np.math.factorial(j-i)) * T[m].clone()**(j-i)
                    #A[m,i+n,j] = (np.math.factorial(j) / np.math.factorial(j-i)) * torch.pow(T[m], (j-i))

    # Assemble block diagonal matrices Q1...M, A1...M
    QM = block_diag(Q)
    AM = block_diag(A)

    # Hessian cost matrix
    H = torch.inverse(AM).T @ QM @ torch.inverse(AM)

    return H, AM


def quad_form(x, Q):
    return x.T @ Q @ x


# Generate Trajectory
n = 4 # order derivative to minimize
N = 2*n # number of polynomial coefficients
W = torch.Tensor([[3], [5], [2]]) # waypoints (first waypoint is at 0)
M = len(W) # number of segments
K = 1e5 # time regularization

# Time segments
#T = torch.ones(M, 1, dtype=torch.float64, requires_grad=True)
# X = torch.ones(M, 1, dtype=torch.float64, requires_grad=True)
# T = torch.nn.functional.relu(X.clone())
T = torch.tensor([[10.0], [10.0], [10.0]], requires_grad=True)

# Hessian cost matrix
H, _ = minSnapCostMat(n, T, W)

# Fixed derivatives
dfix = torch.zeros(M+4, 1, dtype=torch.float64, requires_grad=False)
dfix[4:] = W # Set waypoint constraints

# Free derivatives
dfree = torch.zeros(3*M, 1, dtype=torch.float64, requires_grad=True)

# Form full ordered derivative list
L = [dfix[0:4]] # initial state

for i in range(M): # intermediate states
    L.append(dfix[i+4].unsqueeze(0))
    L.append(dfree[i:i+3])
    if i != M-1: # if not final state, repeat for start of next segment
        L.append(dfix[i+4].unsqueeze(0))
        L.append(dfree[i:i+3])

# Do gradient descent
n_optim_steps = int(1e3)
#optimizer = torch.optim.SGD([dfree, X], 1e-4)
optimizer = torch.optim.Adam([dfree, T], 1e-2)

for ii in range(n_optim_steps):
    optimizer.zero_grad()
    d = torch.cat(L, 0)
    loss = quad_form(d, H) + K * torch.sum(T)
    #loss = quad_form(d, H) 

    print('Step # {}, loss: {}'.format(ii, loss.item()))
    loss.backward(retain_graph=True)

    optimizer.step()
    
print(d)
print(T)

_, AM = minSnapCostMat(n, T, W)

p = (torch.inverse(AM) @ d).detach().numpy().flatten()
Tseg = T.detach().numpy().flatten()
Tstart = np.cumsum(Tseg)

P = np.zeros((N, M))
for i in range(M):
    P[:,i] = np.flip(p[i*N:(i+1)*N])

# Plot generated trajectory
dt = 0.01
for i in range(len(Tseg)):
    Ti = np.arange(0, Tseg[i], dt)
    if i == 0:
        Ts = 0
    else:
        Ts = Tstart[i-1]
    plt.plot(Ts+Ti, np.polyval(P[:,i], Ti))

plt.show()