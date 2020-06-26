import numpy as np
import torch
import matplotlib.pyplot as plt
import plot_utils

from torch.autograd import Function

class Comp_Mat(Function):

    @staticmethod
    def forward(ctx, t):
        A = torch.Tensor([[1, t, t**2, t**3, t**4, t**5, t**6, t**7],
                        [0, 1, 2*t, 3*(t**2), 4*(t**3), 5*(t**4), 6*(t**5), 7*(t**6)],
                        [0, 0, 2, 6*t, 4*3*(t**2), 5*4*(t**3), 6*5*(t**4), 7*6*(t**5)],
                        [0, 0, 0, 6, 4*3*2*(t), 5*4*3*(t**2), 6*5*4*(t**3), 7*6*5*(t**4)]])

        A_back = torch.Tensor([[0, 1, 2*t, 3*(t**2), 4*(t**3), 5*(t**4), 6*(t**5), 7*(t**6)],
                                    [0, 0, 2, 6*t, 4*3*(t**2), 5*4*(t**3), 6*5*(t**4), 7*6*(t**5)],
                                    [0, 0, 0, 6, 4*3*2*(t), 5*4*3*(t**2), 6*5*4*(t**3), 7*6*5*(t**4)],
                                    [0, 0, 0, 0, 4*3*2, 5*4*3*2*(t), 6*5*4*3*(t**2), 7*6*5*4*(t**3)]])
        ctx.save_for_backward(A_back)
        return A

    @staticmethod
    def backward(ctx, grad_output):
        result, = ctx.saved_tensors
        return grad_output * result

class MinSnapTraj():

    def __init__(self, M, W, W_vel = None, idx_vel = None, 
                        W_acc = None, idx_acc = None,
                        W_jer = None, idx_jer = None, time_const = 50):
        
        self.M = M
        self.N = 8
        self.n = 4

        self.time_const = time_const

        self.W = W
        self.idx_pos = np.arange(W.shape[0])

        self.W_vel = W_vel
        self.idx_vel = idx_vel
        self.W_acc = W_acc
        self.idx_acc = idx_acc
        self.W_jer = W_jer
        self.idx_jer = idx_jer

        self.C, self.nf, self.nuf, self.dfix = self.compute_C_deriv_dfix()

        self.QM = torch.zeros(self.M*self.N, self.M*self.N, dtype=torch.float64)
        self.AM = torch.zeros(self.M*self.N, self.M*self.N, dtype=torch.float64)

        self.T = torch.ones(M, 1, dtype=torch.float64, requires_grad=True)
        self.dfree = torch.zeros(self.nuf, dtype=torch.float64, requires_grad=True)

    def forward(self):
        self.QM = self.compute_Q()
        self.AM = self.compute_A()

        C = self.C
        
        H = C @ torch.inverse(self.AM).T @ self.QM @ torch.inverse(self.AM) @ C.T
        d_star = torch.cat((self.dfix, self.dfree))
        return (d_star.T @ H @ d_star) + (self.time_const) * torch.sum(self.T)

    def compute_Q(self):
        Q = torch.zeros((self.M, self.N, self.N), dtype=torch.float64)
        # Q.requires_grad = True
        for m in range(self.M):
            for i in range(self.n, self.N + 1):
                for j in range(self.n, self.N + 1):
                    k = np.arange(0, self.n, dtype=np.float64)
                    Q[m,i-1,j-1] = 2 * np.prod((i-k)*(j-k)) * \
                            self.T[m]**(i+j+1-2*self.n) / (i+j+1-2*self.n)
        return self.block_diag(Q)

    def compute_A(self):
        M = self.M
        N = self.N
        n = self.n
        AM = torch.zeros((M*N, M*N), dtype=torch.float64)
        
        for m in range(M):

            if m == 0:
                AM[:n,:N] = Comp_Mat.apply(0)
                AM[n:2*n, :N] = Comp_Mat.apply(self.T[m])
            else:
                AM[m*N:m*N+n, (m-1)*N:m*N] = Comp_Mat.apply(self.T[m-1])
                AM[m*N:m*N+n, m*N: (m+1)*N] = -Comp_Mat.apply(0)
                AM[m*N+n:m*N+2*n, m*N: (m+1)*N] = Comp_Mat.apply(self.T[m])

        return AM

    def compute_C_deriv_dfix(self):

        M = self.M
        N = self.N
        n = self.n

        fixed_deriv = []

        pos_fixed_idx = self.find_deriv_idx(self.idx_pos, M, N, n, 0)
        vel_fixed_idx = self.find_deriv_idx(self.idx_vel, M, N, n, 1)
        acc_fixed_idx = self.find_deriv_idx(self.idx_acc, M, N, n, 2)
        jer_fixed_idx = self.find_deriv_idx(self.idx_jer, M, N, n, 3)
        fixed_deriv.append(pos_fixed_idx)
        fixed_deriv.append(vel_fixed_idx)
        fixed_deriv.append(acc_fixed_idx)
        fixed_deriv.append(jer_fixed_idx)

        fixed_deriv.append(self.find_zero_idx(M, N, n))

        fixed_deriv = [item for sublist in fixed_deriv for item in sublist]
        
        unfixed_deriv = self.find_unfixed_deriv(fixed_deriv, M, N)

        C = torch.zeros((M*N, M*N), dtype=torch.float64)
        nf = len(fixed_deriv)
        nuf = len(unfixed_deriv)

        for i in range(nf):
            C[i, fixed_deriv[i]] = 1
        for i in range(nuf):
            C[i+nf, unfixed_deriv[i]] = 1
        
        d = torch.zeros((M*N,), dtype=torch.float64)
        d[pos_fixed_idx] = torch.flatten(self.W)
        if self.idx_vel is not None:
            d[vel_fixed_idx] = torch.flatten(self.W_vel)
        if self.idx_acc is not None:
            d[acc_fixed_idx] = torch.flatten(self.W_acc)
        if self.idx_jer is not None:
            d[jer_fixed_idx] = torch.flatten(self.W_jer)

        dfix = (C @ d)[:nf]

        return C, nf, nuf, dfix

    ### 
    # utils

    def get_traj(self):
        N = self.N
        M = self.M
        C = self.C
        d = C.T @ torch.cat((self.dfix.detach(), self.dfree.detach()))
        p = (torch.inverse(self.AM) @ d).detach().numpy().flatten()
        Tseg = (self.T).detach().numpy().flatten()
        P = np.zeros((N, M))
        for i in range(M):
            P[:,i] = np.flip(p[i*N:(i+1)*N])
        
        return P, Tseg

    def block_diag(self, m):
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
        eye = self.attach_dim(torch.eye(n).unsqueeze(-2), d - 3, 1)
        return (m2 * eye).reshape(
            siz0 + torch.Size(torch.tensor(siz1) * n)
        )

    def attach_dim(self, v, n_dim_to_prepend=0, n_dim_to_append=0):
        return v.reshape(
            torch.Size([1] * n_dim_to_prepend)
            + v.shape
            + torch.Size([1] * n_dim_to_append))

    def getA_torch(self, t):
        '''
        assumes snap
        '''
        A = torch.Tensor([[1, t, t**2, t**3, t**4, t**5, t**6, t**7],
                    [0, 1, 2*t, 3*(t**2), 4*(t**3), 5*(t**4), 6*(t**5), 7*(t**6)],
                    [0, 0, 2, 6*t, 4*3*(t**2), 5*4*(t**3), 6*5*(t**4), 7*6*(t**5)],
                    [0, 0, 0, 6, 4*3*2*(t), 5*4*3*(t**2), 6*5*4*(t**3), 7*6*5*(t**4)]])
        return A

    def find_deriv_idx(self, idx_list, M, N, n, this_num):
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

    def find_zero_idx(self, M, N, n):
        
        this_list = []
        
        for m in range(M):
            if m == 0:
                continue

            for idx in range(n):
                this_list.append(m*N + idx)

        return this_list

    def find_unfixed_deriv(self, fixed_deriv, M, N):

        arr1 = np.array(fixed_deriv)
        arr2 = np.arange(N * M)

        this_list = list(np.setdiff1d(arr2, arr1, assume_unique=True))
        return this_list

def learn_time_and_dfree(model, optimizer, epochs, print_every = 50):
    for e in range(epochs):
        score = model.forward()
        optimizer.zero_grad()
        score.backward()
        optimizer.step()
        if e % print_every == 0:
            print('Iteration %d, loss = %.4f' % (e, score.item()))

# M = 2
# W = torch.Tensor([[3], [5], [2]]).double()

# learning_rate = 1e-1
# model = MinSnapTraj(M = M, W = W, time_const = 5000)
# optimizer = torch.optim.Adam([model.T, model.dfree], lr=learning_rate)

# learn_time_and_dfree(model, optimizer, 10000)

# print(model.T)
# print(model.dfree)

# P, T = model.get_traj()

# plot_utils.plot_single_traj(P, T, 0.01)

# plt.show()