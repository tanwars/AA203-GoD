'''
Just a tester script - a sandbox. 
Is not important for the functioning of the package.
'''

import airsim
# import types

import numpy as np
from scipy.signal import cont2discrete
from scipy import linalg as la
from drone_util import get_drone_state, not_reached, get_throttle, bound_control

from models import linear_quad_model, acados_linear_quad_model, \
                    acados_nonlinear_quad_model
from acados_ocp_problem_form import create_ocp_solver
from acados_template import AcadosOcp, AcadosOcpSolver

from minsnap_trajgen import minSnapTG as mstg
from quadrotor import Quadrotor, Trajectory

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

import time

# some parameters
g = 9.81                    # acc. due to gravity
dt = 0.1                    # time step
n = 9                       # state dimension
m = 4                       # control input dimension
kq = 1                      # parameter to tune the Q matrix Q = kq * I 
kr = 1#1 works for linear case                # parameter to tune the R matrix R = kr * I

rotate_quad = False  # set to true if you want yaw to change between waypoints
mode = 'nonlinear'
setGains = False
(kp_roll, ki_roll, kd_roll) = (1,0,0.1)
(kp_pitch, ki_pitch, kd_pitch) = (1,0,0.1)
(kp_yaw, ki_yaw, kd_yaw) = (10,0,0.1)

Tf = 0.5 # time horizon for MPC
dist_check = 15.0           # distance to waypoint to say waypoint reached
throttle_const = 0.59375    # constant mapping u4 to throttle(t): t = tc * u4 
max_abs_roll_rate = 50.0#50.0     # clamping roll rate
max_abs_pitch_rate = 50.0#50.0    # clamping pitch rate
max_abs_yaw_rate = 50.0#50.0      # clamping yaw rate
max_iter = 400              # maximum iterations to time out of the loop
ue = np.array([0,0,0,g])    # nominal control


wpfile = 'waypts_test.csv'  # waypoint file
minimizer = 4 # 4 for snap, 3 for jerk
degree = 2 * minimizer
time_dilation = 6 #works

### 
# set gain parameters for the low level controller

roll_rate_gains = airsim.PIDGains(kp_roll, ki_roll, kd_roll)
pitch_rate_gains = airsim.PIDGains(kp_pitch, ki_pitch, kd_pitch)
yaw_rate_gains = airsim.PIDGains(kp_yaw, ki_yaw, kd_yaw)
angle_rate_gains = airsim.AngleRateControllerGains(roll_rate_gains, pitch_rate_gains,
                                                yaw_rate_gains)

###
#  get trajectory
###

t_kf = time_dilation * np.array([1,1,1,1,1,1,1,1], dtype=np.float64) # time of flight
wpts = np.loadtxt(wpfile)
num_wpts = wpts.shape[0]

# just some manipulations for testing
z_offset = -20.0
num_wpts = (t_kf.size) + 1
wpts[:,2] += z_offset
wpts_test = wpts[:num_wpts,:].copy()
psi_test = np.zeros((num_wpts,))
if rotate_quad == True:
    for i in range(1,psi_test.shape[0]):
        psi_test[i] = np.arctan2(wpts_test[i,1] - wpts_test[i-1,1],
                                wpts_test[i,0] - wpts_test[i-1,0])


P = [None] * 4
for i in range(3):
    P[i] = mstg(degree, minimizer, t_kf, wpts_test[:,i])
P[3] = mstg(degree, minimizer, t_kf, psi_test)
traj = np.stack(P,axis = 2)

trajectory = Trajectory(traj, np.cumsum(t_kf))

### 
# initialize quadrotor model
q = Quadrotor(dt = dt)
init_pos, init_control = q.setStateAtNomTime(trajectory, 0)
total_steps = int(np.floor((np.cumsum(t_kf))[-1]/ dt))

### 
# initialize the controller
if mode == 'nonlinear':
    print('--------------------')
    print('Using Non-linear model')
    print('--------------------')
    model = acados_nonlinear_quad_model(9, g = g)
else:
    print('--------------------')
    print('Using linear model')
    print('--------------------')
    model = acados_linear_quad_model(9, g = g)

ocp_solver = create_ocp_solver(model, Tf, dt, kq, kr, n, m, 
                    max_abs_roll_rate, max_abs_pitch_rate,
                    max_abs_yaw_rate, init_pos[:n])
N = int(Tf/dt)


###
#  connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
if setGains == True:
    client.setAngleRateControllerGains(angle_rate_gains)

###
#  Take off and move to an initial starting point
airsim.wait_key('Press any key to takeoff')
client.takeoffAsync().join()
# client.moveToPositionAsync(0, 0, -20, 5).join()
client.moveToPositionAsync(init_pos[0], init_pos[1], init_pos[2], 5).join()
airsim.wait_key('Now we start our task!')

###
# Logging params
all_controls = np.zeros((total_steps-1,m))

# calculate the nominal path
traj_ptx = []
traj_pty = []
traj_ptz = []
traj_ptp = []
# dt = 0.01
for num in range(len(t_kf)):
    T = t_kf[num]
    x = np.poly1d(traj[:,num,0])
    y = np.poly1d(traj[:,num,1])
    z = np.poly1d(traj[:,num,2])
    psi = np.poly1d(traj[:,num,3])
    for i in range(int(T/dt)):
        t = i * dt
        traj_ptx.append(x(t))
        traj_pty.append(y(t))
        traj_ptz.append(z(t))
        traj_ptp.append(psi(t))

all_est_path = np.zeros((total_steps-1,n))


print('-------- Executing the path --------')

tt = 0
# for i in range(total_steps-1):
# while tt < (np.cumsum(t_kf))[-1] :
for i in range(5):
    # q.setSystemTime(i * dt)
    
    start_time = time.time()
    # tt = time.time() - start_time  # current time
    tt = i * dt
    for num1 in range(N):
        t = tt + num1 * dt
        full_state, _ = q.getStateAtNomTime(trajectory, t)
        yref  = np.hstack((full_state[:n], ue))
        ocp_solver.set(num1, "yref", yref)
    t = tt + N * dt
    full_state, _ = q.getStateAtNomTime(trajectory, t)
    yref_e = full_state[:n].copy()
    ocp_solver.set(N, "yref", yref_e)

    status = ocp_solver.solve()
    if status != 0:
        raise Exception('acados returned status {}. Exiting.'.format(status))
	
    print("OCP--- %s seconds ---" % (time.time() - start_time))

    start_time = time.time()

    u = ocp_solver.get(0, "u")
    est_x = ocp_solver.get(1, "x")

    all_controls[i,:] = u.copy()
    all_est_path[i,:] = est_x.copy()

    rotmat_u = np.array([[1,0,0],[0,-1,0],[0,0,-1]])
    u[0:3] = rotmat_u @ u[0:3]

    client.moveByAngleRatesThrottleAsync(u[0], u[1], u[2], u[3], dt).join()
    x = get_drone_state((client.getMultirotorState()).kinematics_estimated, n)

    ocp_solver.set(0, "lbx", x)
    ocp_solver.set(0, "ubx", x)
    print("exec--- %s seconds ---" % (time.time() - start_time))

print('-------- Done Executing --------')

airsim.wait_key('Phew!')
client.armDisarm(False)
client.reset()

# that's enough fun for now. let's quit cleanly
client.enableApiControl(False)

### 
# plotting

plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(traj_ptx, traj_pty, traj_ptz, 'gray')
ax.scatter3D(wpts_test[:,0],wpts_test[:,1],wpts_test[:,2], color = 'g')
ax.scatter3D(
    all_est_path[:,0], all_est_path[:,1], 
    all_est_path[:,2], color = 'b')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_zlim(0,-80)

plt.figure()
plt.plot(all_controls[:,0], label = 'u0')
plt.plot(all_controls[:,1], label = 'u1')
plt.plot(all_controls[:,2], label = 'u2')
plt.plot(all_controls[:,3], label = 'u3')
plt.legend()
plt.show()
