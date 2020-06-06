import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.integrate import ode

class Quadrotor():
    """
    Quadrotor non-linear model, and states from differentially flat output
    """
    def __init__(self, m = 1, kf = 1e-5, km = 1e-5, Jx = 1, Jy = 1, Jz = 2, 
                       l = 1, dt = 0.01, fidelity = 12):
        self.m = m
        self.kf = kf
        self.km = km
        self.J = np.diag(np.array([Jx,Jy,Jz]))
        self.l = l
        self.dt = dt
        self.g = 9.81
        self.fidelity = fidelity
        self.time = 0
        assert (fidelity == 12 or fidelity == 9), "fidelity should be 12 or 9"
        if fidelity == 12:
            self.state = np.zeros((12,))
        elif fidelity == 9:
            self.state = np.zeros((9,))
        self.state_dot = np.zeros_like(self.state)

    def setState(self, x):
        assert x.size == self.fidelity, "fidelity is different. Cannot set state"
        self.state = x.copy()

    def setSystemTime(self, t):
        self.time = t

    def getState(self):
        """
        returns the current state estimate
        """
        return self.state

    def EulStepNlDyn(self, u):
        """
        Non-linear model in X config.
        control u is thrust, torques - assumes it in x config.
        outputs state at next time state and derivative at current
        """
        x_dot = self.nlDyn(u, self.state)
        self.state += self.dt * x_dot
        self.time += self.dt
        return self.state, x_dot

    def nlDyn(self, u, x):
        """
        Non-linear model in X config.
        control u is thrust, torques - assumes it in x config.
        outputs state at next time state and derivative at current
        """

        self.state = x
        x_dot = np.zeros_like(x)
        g = self.g
        m = self.m
        J = self.J

        phi = x[3]
        theta = x[4]
        psi = x[5]
        omega = x[9:12]

        x_dot[0:3] = x[6:9].copy()
        
        r1 = np.array([ [1,0,0],
                        [0,np.cos(phi),-np.sin(phi)],
                        [0,np.sin(phi),np.cos(phi)]])
        r2 = np.array([ [np.cos(theta),0,np.sin(theta)],
                        [0,1,0],
                        [-np.sin(theta),0,np.cos(theta)]])               
        r3 = np.array([ [np.cos(psi),-np.sin(psi),0],
                        [np.sin(psi),np.cos(psi),0],
                        [0,0,1]])

        Tau = np.array([[np.cos(theta), 0, -np.sin(theta)],
                        [0, 1, np.sin(phi) * np.cos(theta)],
                        [np.sin(theta), 0, np.cos(phi) * np.cos(theta)]])
        x_dot[3:6] = np.linalg.inv(Tau) @ omega

        rot_arr = r3 @ r1 @ r2 @ np.array([0,0,1])
        x_dot[6:9] = (-(u[0]/m) * rot_arr) + np.array([0,0,g])

        x_dot[9:12] = np.linalg.inv(J) @ (u[1:4] - np.cross(omega, J @ omega))

        self.state_dot = x_dot.copy()
        return x_dot

    def diffFlatStatesInputs(self, traj):
        """
        Computes the state and input from differentially flat output
        outputs: 
            desired control for each time step dt upto t_kf[-1]
            desired quadrotor state for each time step dt upto t_kf[-1] 
        """

        # find total number of timesteps
        t_kf = traj.getTkf()
        total_steps = int(np.floor((t_kf[-1]/ self.dt)))

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

            # find state and control
            x = np.zeros((self.fidelity,))
            u = np.zeros((4,))
            psi = (traj.sigma(t))[3]
            psi_dot = (traj.sigma(t,1))[3]
            psi_ddot = (traj.sigma(t,2))[3]
            sigma_pos = (traj.sigma(t))[0:3]
            sigma_vel = (traj.sigma(t,1))[0:3]
            sigma_acc = (traj.sigma(t,2))[0:3]
            sigma_jer = (traj.sigma(t,3))[0:3]
            sigma_sna = (traj.sigma(t,4))[0:3]
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

            omega_nw = np.array([x[9],x[10],x[11]])

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
            omega_dot_nw = np.array([alpha1,alpha2,alpha3])
            
            # possibly minus sign
            u_vec = J @ omega_dot_nw + np.cross(omega_nw, J @ omega_nw)
            u[1:4] = u_vec

            x_all[:,time_step] = x
            u_all[:,time_step] = u

        return x_all, u_all

    def getStateAtNomTime(self, traj, t):
        '''
        returns the nominal state of the quadrotor along a trajectory at time t
        '''
        
        psi_t = traj.sigma(t)[3]
        psi_dot_t = traj.sigma(t,1)[3]
        pos_t = traj.sigma(t)[0:3]
        vel_t = traj.sigma(t,1)[0:3]
        acc_t = traj.sigma(t,2)[0:3]
        jer_t = traj.sigma(t,3)[0:3]
        mass = self.m
        g = self.g

        out_state = np.zeros_like(self.state)
        zw = np.array([0,0,1])

        out_state[0:3] = pos_t.copy()
        out_state[6:9] = vel_t.copy()
        out_state[5] = psi_t

        Fn = mass * acc_t - mass * g * zw
        u0_ff = - np.dot(zw, Fn)

        zn = - Fn / np.linalg.norm(Fn)
        ys = np.array([-np.sin(psi_t), np.cos(psi_t), 0])
        xn = np.cross(ys, zn) / np.linalg.norm(np.cross(ys, zn))
        yn = np.cross(zn, xn)

        R_mat_np = np.hstack(
            (np.reshape(xn,(3,1)),np.reshape(yn,(3,1)),np.reshape(zn,(3,1))))
        R_mat = R.from_matrix(R_mat_np)

        eul_angles = R_mat.as_euler('zxy') ## this might be problematic.
        # consider manually doing it.
        out_state[3] = eul_angles[1]
        out_state[4] = eul_angles[2]

        h_omega = (mass / u0_ff) * ((np.dot(jer_t, zn)) * zn - jer_t)
        out_state[9] = -np.dot(h_omega, yn)
        out_state[10] = np.dot(h_omega, xn)
        out_state[11] = psi_dot_t * np.dot(zw, zn)
        return out_state.copy()

    def setStateAtNomTime(self, traj, t):
        '''
        Sets the quadrotor state to the nominal path state at time t
        '''
        out_state = self.getStateAtNomTime(traj, t)
        self.setState(out_state)
        return out_state.copy()


    def PDController(self, traj):
        """
        looks at the current state and nominal differentially flat path to 
        compute the PD control to be applied.
        """

        # state params
        x = (self.state).copy()  # assume perfectly measurable system
        t = self.time
        mass = self.m
        g = self.g
        J = self.J

        pos = x[0:3]
        phi = x[3]
        theta = x[4]
        psi = x[5]
        vel = x[6:9]
        omega = x[9:12]
        
        r1 = np.array([ [1,0,0],
                        [0,np.cos(phi),-np.sin(phi)],
                        [0,np.sin(phi),np.cos(phi)]])
        r2 = np.array([ [np.cos(theta),0,np.sin(theta)],
                        [0,1,0],
                        [-np.sin(theta),0,np.cos(theta)]])               
        r3 = np.array([ [np.cos(psi),-np.sin(psi),0],
                        [np.sin(psi),np.cos(psi),0],
                        [0,0,1]])

        Rb = r3 @ r1 @ r2
        
        # trajectory params
        psi_t = traj.sigma(t)[3]
        psi_dot_t = traj.sigma(t,1)[3]
        psi_ddot_t = traj.sigma(t,2)[3]
        pos_t = traj.sigma(t)[0:3]
        vel_t = traj.sigma(t,1)[0:3]
        acc_t = traj.sigma(t,2)[0:3]
        jer_t = traj.sigma(t,3)[0:3]
        sna_t = traj.sigma(t,4)[0:3]

        # control 
        u = np.zeros((4,))

        # gains
        (gxy, gz, gxyv, gzv, gxyr, gzr, gxyw, gzw) = \
            (10 , 10 , 1   , 1  , 1000   , 100  , 10   , 1)
        
        Ke = np.eye(3) * np.array([gxy,gxy,gz])
        Kv = np.eye(3) * np.array([gxyv,gxyv,gzv])
        Kr = np.eye(3) * np.array([gxyr,gxyr,gzr])
        Kw = np.eye(3) * np.array([gxyw,gxyw,gzw])

        zw = np.array([0,0,1])

        e_x = pos_t - pos
        e_v = vel_t - vel
        Fdes = mass * acc_t - mass * g * zw + Ke @ e_x + Kv @ e_v
        u[0] = - np.dot(Rb[:,2], Fdes)

        zd = - Fdes / np.linalg.norm(Fdes)
        ys = np.array([-np.sin(psi_t), np.cos(psi_t), 0])
        xd = np.cross(ys, zd) / np.linalg.norm(np.cross(ys, zd))
        yd = np.cross(zd, xd)

        Rdes = np.hstack(
            (np.reshape(xd,(3,1)),np.reshape(yd,(3,1)),np.reshape(zd,(3,1))))

        h_omega_d = (mass / u[0]) * ((np.dot(jer_t, zd)) * zd - jer_t)
        p_d = -np.dot(h_omega_d, yd)
        q_d = np.dot(h_omega_d, xd)
        r_d = psi_dot_t * np.dot(zw, zd)
        omega_d = np.array([p_d, q_d, r_d])

        skew_mat = 0.5 * ( np.transpose(Rb) @ Rdes - np.transpose(Rdes) @ Rb)
        e_r = np.array([skew_mat[2,1], skew_mat[0,2], skew_mat[1,0]])
        e_w = (np.transpose(Rb) @ Rdes @ omega_d) - omega

        Fn = mass * acc_t - mass * g * zw
        u0_ff = - np.dot(Rb[:,2], Fn)
        zn = - Fn / np.linalg.norm(Fn)
        xn = np.cross(ys, zn) / np.linalg.norm(np.cross(ys, zn))
        yn = np.cross(zn, xn)

        Rn = np.hstack(
            (np.reshape(xn,(3,1)),np.reshape(yn,(3,1)),np.reshape(zn,(3,1))))

        h_omega = (mass / u0_ff) * ((np.dot(jer_t, zn)) * zn - jer_t)
        p_t = -np.dot(h_omega, yn)
        q_t = np.dot(h_omega, xn)
        r_t = psi_dot_t * np.dot(zw, zn)

        omega_nw = np.array([p_t,q_t,r_t])

        u0_ff_dot = - mass * np.dot(jer_t, zn)
        u0_ff_ddot = -np.dot(
            (mass * sna_t + np.cross(omega_nw, np.cross(omega_nw, zn))),
            zn)

        h_alpha = (-1.0/u0_ff) * (
            mass * sna_t + u0_ff_ddot * zn + \
                2.0 * u0_ff_dot * np.cross(omega_nw,zn) + \
                    np.cross(omega_nw, np.cross(omega_nw, zn)))
        
        alpha1 = - np.dot(h_alpha, yn)
        alpha2 = np.dot(h_alpha, xn)
        alpha3 = np.dot((psi_ddot_t * zn - psi_dot_t * h_omega), zw)
        omega_dot_nw = np.array([alpha1,alpha2,alpha3])

        u_vec = J @ (np.transpose(Rb) @ Rn @ omega_dot_nw - 
                    np.cross(omega, np.transpose(Rb) @ Rn @ omega_nw)) + \
                np.cross(omega, J @ omega) + Kr @ e_r + Kw @ e_w

        u[1:4] = u_vec
        return u

    def solveSystem(self, init_pos, T, traj):
        '''
        does ODE45 integration to progress the system along a trajectory
        '''

        t0 = 0
        self.state = init_pos

        def controlSystem(t, x, trajectory):
            self.setSystemTime(t)
            u = self.PDController(trajectory)
            return self.nlDyn(u,x)
        
        backend = 'dopri5'
        nsteps = 5000
        atol = 1e-8
        rtol = 1e-8
        solver = ode(controlSystem).set_integrator(backend, nsteps = nsteps, 
                                                    atol = atol, rtol = rtol)
        total_steps = int(np.floor((T/ self.dt)))

        all_states = []
        all_t = []
        for time_step in range(total_steps):
            state = []
            time = []
            t_init = t0 + time_step * self.dt
            def solout(t, y):
                time.append(t)
                state.append(y.copy()) 
            solver.set_solout(solout)
            solver.set_initial_value(self.state, t_init).set_f_params(traj)
            solver.integrate(t_init + self.dt)
            self.state = state[-1]
            all_states.append(state)
            all_t.append(time)

        return np.hstack(all_t), np.transpose(np.vstack(all_states))

class Trajectory():
    """
    trajectories to store and access classes.
    """

    def __init__(self, sigma_coeffs, t_kf):
        """
        sigma_coeffs is shaped (n, m, 4)
        t_kf is shaped (m,)
        """
        self.sigma_coeffs = sigma_coeffs
        self.t_kf = t_kf

    def getTkf(self):
        return self.t_kf

    def sigma(self, t, order = 0):
        j = np.searchsorted(self.t_kf,t)
        sigma_arr = [None] * 4
        sigma_ret = [None] * 4
        for idx in range(4):
            sigma_arr[idx] = np.poly1d(self.sigma_coeffs[:,j,idx])
            if order != 0:
                sigma_ret[idx] = np.polyder(sigma_arr[idx], order)
        
        if order == 0:
            return np.array([sigma_arr[0](t-j), 
                             sigma_arr[1](t-j), 
                             sigma_arr[2](t-j),
                             sigma_arr[3](t-j)])
        else:
            return np.array([sigma_ret[0](t-j), 
                             sigma_ret[1](t-j), 
                             sigma_ret[2](t-j),
                             sigma_ret[3](t-j)])
    