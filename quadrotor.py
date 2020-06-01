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

        rot_arr = np.array([np.cos(psi) * np.sin(theta) * np.cos(phi) + np.sin(psi) * np.sin(phi),
                            np.sin(psi) * np.sin(theta) * np.cos(phi) - np.cos(psi) * np.sin(phi),
                            np.cos(theta) * np.cos(phi)])
        x_dot[6:9] = -(u[0]/m) * rot_arr + np.array([0,0,g])

        x_dot[9:12] = np.linalg.inv(J) @ (u[1:4] - np.cross(omega, J @ omega))

        self.state += self.dt * x_dot
        return (self.state, x_dot)
