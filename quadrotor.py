import numpy as np

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
        
        Tau = np.array([[1,0, - np.sin(theta)],
                        [0,np.cos(phi), np.sin(phi) * np.cos(theta)],
                        [0,-np.sin(phi), np.cos(phi) * np.cos(theta)]])
        x_dot[3:6] = np.linalg.inv(Tau) @ omega

        rot_arr = np.array(
            [np.cos(psi) * np.sin(theta) * np.cos(phi) + np.sin(psi) * np.sin(phi),
            np.sin(psi) * np.sin(theta) * np.cos(phi) - np.cos(psi) * np.sin(phi),
            np.cos(theta) * np.cos(phi)])
        x_dot[6:9] = -(u[0]/m) * rot_arr + np.array([0,0,g])

        x_dot[9:12] = np.linalg.inv(J) @ (u[1:4] - np.cross(omega, J @ omega))

        self.state += self.dt * x_dot
        return (self.state, x_dot)

    def diffFlatStatesInputs(self, sigma_coeffs, t_kf):
        """
        Computes the state and input from differentially flat output
        sigma_coeffs is shaped (n, m, 4)
        t_kf is shaped (m,)
        outputs: 
            desired control for each time step dt upto t_kf[-1]
            desired quadrotor state for each time step dt upto t_kf[-1] 
        """
        (n,m,_) = sigma_coeffs.shape
        sigma = [None] * 4
        sigma1d = [None] * 4
        sigma2d = [None] * 4
        sigma3d = [None] * 4
        sigma4d = [None] * 4

        # find total number of timesteps
        total_steps = int(np.floor(t_kf[-1]/ self.dt))

        # initialize x and u and other params
        x = np.zeros((self.fidelity,))
        u = np.zeros((4,))
        g = self.g

        # for each timestep
        for time_step in range(total_steps):
            # find which piece we are in
            t = time_step * self.dt
            j = np.searchsorted(t_kf,t)
            
            # compute derivatives of sigma(t) - can be optimized to be called
            # once per j change
            for idx in range(4):
                sigma[idx] = np.poly1d(sigma_coeffs[:,j,idx])
                sigma1d[idx] = np.polyder(sigma[idx])
                sigma2d[idx] = np.polyder(sigma1d[idx])
                sigma3d[idx] = np.polyder(sigma2d[idx])
                sigma4d[idx] = np.polyder(sigma3d[idx])

            # find state 
            x[0:3] = np.array([sigma[0](t), sigma[1](t), sigma[2](t)])
            x[6:9] = np.array([sigma1d[0](t), sigma1d[1](t), sigma1d[2](t)])
            x[5] = sigma[3](t)

            # find control 

q = Quadrotor()

c = np.reshape(np.array([[1,1,1,1],[2,2,2,1]]),(2,1,4))
t_kf = [0.1]
q.diffFlatStatesInputs(c,t_kf)
