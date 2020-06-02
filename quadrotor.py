import numpy as np
from scipy.spatial.transform import Rotation as R

class Quadrotor():
    """
    Quadrotor non-linear model, and states from differentially flat output
    """
    def __init__(self, m = 1, kf = 1e-5, km = 1e-5, Jx = 1, Jy = 1, Jz = 1, 
                       l = 1, dt = 0.01, fidelity = 12):
        self.m = m
        self.kf = kf
        self.km = km
        self.J = np.diag(np.array([Jx,Jy,Jz]))
        self.l = l
        self.dt = dt
        self.g = 9.81
        self.fidelity = fidelity
        assert (fidelity == 12 or fidelity == 9), "fidelity should be 12 or 9"
        if fidelity == 12:
            self.state = np.zeros((12,))
        elif fidelity == 9:
            self.state = np.zeros((9,))

    def setState(self, x):
        assert x.size == self.fidelity, "fidelity is different. Cannot set state"
        self.state = x

    def getState(self):
        """
        returns the current state estimate
        """
        return self.state

    def stepNlDyn(self, u):
        """
        Non-linear model in X config.
        control u is thrust, torques - assumes it in x config.
        outputs state at next time state and derivative at current
        """
        x = self.state
        x_dot = np.zeros_like(x)
        g = self.g
        m = self.m
        J = self.J

        phi = x[3]
        theta = x[4]
        psi = x[5]
        omega = x[9:12]

        x_dot[0:3] = x[6:9]
        
        r1 = np.array([ [1,0,0],
                        [0,np.cos(phi),-np.sin(phi)],
                        [0,np.sin(phi),np.cos(phi)]])
        r2 = np.array([ [np.cos(theta),0,np.sin(theta)],
                        [0,1,0],
                        [-np.sin(theta),0,np.cos(theta)]])               
        r3 = np.array([ [np.cos(psi),-np.sin(psi),0],
                        [np.sin(psi),np.cos(psi),0],
                        [0,0,1]])

        col1 = np.reshape(r2[:,0],(3,1))
        col2 = np.reshape(np.array([0,1,0]),(3,1))
        col3 = np.reshape((r2 @ r1)[:,2],(3,1))
        Tau = np.hstack((col1,col2,col3))
        x_dot[3:6] = np.linalg.inv(Tau) @ omega

        rot_arr = r3 @ r1 @ r2 @ np.array([0,0,1])
        x_dot[6:9] = (-(u[0]/m) * rot_arr) + np.array([0,0,g])

        x_dot[9:12] = np.linalg.inv(J) @ (u[1:4] - np.cross(omega, J @ omega))

        self.state += self.dt * x_dot
        return self.state, x_dot

    def diffFlatStatesInputs(self, sigma_coeffs, t_kf):
        """
        Computes the state and input from differentially flat output
        sigma_coeffs is shaped (n, m, 4)
        t_kf is shaped (m,)
        outputs: 
            desired control for each time step dt upto t_kf[-1]
            desired quadrotor state for each time step dt upto t_kf[-1] 
        """
        sigma = [None] * 4
        sigma1d = [None] * 4
        sigma2d = [None] * 4
        sigma3d = [None] * 4
        sigma4d = [None] * 4

        # find total number of timesteps
        total_steps = int(np.floor(t_kf[-1]/ self.dt))

        # initialize x and u and other params
        x_all = np.zeros((self.fidelity,total_steps))
        u_all = np.zeros((4,total_steps))
        g = self.g
        mass = self.m
        J = self.J

        # for each timestep
        for time_step in range(total_steps):
            # find which piece we are in
            t = time_step * self.dt
            j = np.searchsorted(t_kf,t)
            
            # compute derivatives of sigma(t) - can be sped up/optimized 
            # to be called once per j change
            for idx in range(4):
                sigma[idx] = np.poly1d(sigma_coeffs[:,j,idx])
                sigma1d[idx] = np.polyder(sigma[idx])
                sigma2d[idx] = np.polyder(sigma1d[idx])
                sigma3d[idx] = np.polyder(sigma2d[idx])
                sigma4d[idx] = np.polyder(sigma3d[idx])

            # find state
            # find control  
            x = np.zeros((self.fidelity,))
            u = np.zeros((4,))
            psi = sigma[3](t)
            psi_dot = sigma1d[3](t)
            psi_ddot = sigma2d[3](t)
            sigma_pos = np.array([sigma[0](t), sigma[1](t), sigma[2](t)])
            sigma_vel = np.array([sigma1d[0](t), sigma1d[1](t), sigma1d[2](t)])
            sigma_acc = np.array([sigma2d[0](t), sigma2d[1](t), sigma2d[2](t)])
            sigma_jer = np.array([sigma3d[0](t), sigma3d[1](t), sigma3d[2](t)])
            sigma_sna = np.array([sigma4d[0](t), sigma4d[1](t), sigma4d[2](t)])
            zw = np.array([0,0,1])

            x[0:3] = sigma_pos
            x[6:9] = sigma_vel
            x[5] = psi

            a = sigma_acc
            Fn = mass * a - mass * g * zw
            u[0] = np.linalg.norm(Fn)

            zn = - Fn / np.linalg.norm(Fn)
            ys = np.array([-np.sin(psi), np.cos(psi), 0])
            xn = np.cross(ys, zn) / np.linalg.norm(np.cross(ys, zn))
            yn = np.cross(zn, xn)

            R_mat = R.from_matrix(np.hstack(
                (np.reshape(xn,(3,1)),np.reshape(yn,(3,1)),np.reshape(zn,(3,1)))))

            eul_angles = R_mat.as_euler('zxy') ## this might be problematic.
            # consider manually doing it.
            x[3] = eul_angles[1]
            x[4] = eul_angles[2]

            h_omega = (mass / u[0]) * ((np.dot(sigma_jer, zn)) * zn - sigma_jer)
            x[9] = -np.dot(h_omega, yn)
            x[10] = np.dot(h_omega, xn)
            x[11] = psi_dot * np.dot(zw, zn)

            omega_nw = x[9] * xn + x[10] * yn + x[11] * zn

            u1_dot = - mass * np.dot(sigma_jer, zn)
            u1_ddot = -np.dot(
                (mass * sigma_sna + np.cross(omega_nw, np.cross(omega_nw, zn))),
                zn)

            h_alpha = (-1.0/u[0]) * (
                mass * sigma_sna + u1_ddot * zn + \
                    2.0 * u1_dot * np.cross(omega_nw,zn) + \
                        np.cross(omega_nw, np.cross(omega_nw, zn)))
            
            alpha1 = - np.dot(h_alpha, yn)
            alpha2 = np.dot(h_alpha, xn)
            alpha3 = np.dot((psi_ddot * zn - psi_dot * h_omega), zw)
            omega_dot_nw = alpha1 * xn + alpha2 * yn + alpha3 * zn

            u_vec = J @ omega_dot_nw + np.cross(omega_nw, J @ omega_nw)
            u[1:4] = u_vec

            x_all[:,time_step] = x
            u_all[:,time_step] = u

        return x_all, u_all
