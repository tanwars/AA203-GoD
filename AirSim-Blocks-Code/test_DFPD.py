'''
Just a tester script - a sandbox. 
Is not important for the functioning of the package.
'''

from quadrotor import Quadrotor, Trajectory
import numpy as np

import airsim
from drone_util import get_drone_state, not_reached, get_throttle, bound_control
from minsnap_trajgen import minSnapTG as mstg

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

dt = 0.01
throttle_const = 0.59375    # constant mapping u4 to throttle(t): t = tc * u4
g = 9.81

q = Quadrotor(dt = dt)
# q.setGains(10, 1, 10, 10, 0.5, 1, 1, 1)
q.setGains(0.1, 0.1, 0, 0, 0.1, 0.1, 0, 0)
# q.setGains(100 , 1 , 10   , 1 , 100   , 100  , 10   , 10)

minimizer = 4 # 4 for snap, 3 for jerk
degree = 2 * minimizer

time_dilation = 10
t_kf = time_dilation * np.array([1,2,2], dtype=np.float64) # time of flight

print(t_kf)
print(np.cumsum(t_kf))
wpfile = 'waypts_test.csv'  # waypoint file
wpts = np.loadtxt(wpfile)
num_wpts = wpts.shape[0]

print(wpts)

# just some manipulations for testing
z_offset = -10.0
num_wp = (t_kf.size) + 1
wpts[:,2] += z_offset
wpts_test = wpts[:num_wp,:].copy()
psi_test = np.zeros((num_wp,))

# gen complete path
P = [None] * 4
for i in range(3):
    P[i] = mstg(degree, minimizer, t_kf, wpts_test[:,i])
P[3] = mstg(degree, minimizer, t_kf, psi_test)

# print(len(P))
# print(P[0].shape)
# print(np.stack(P,axis = 2).shape)

traj = np.stack(P,axis = 2)

print(traj.shape)
# traj = np.array([[[0,0,0,0]],[[0,0,0,0]],[[0,1,0,0]],
#                 [[1,0,0,0]],[[1,1,1,0]],[[-2,-2,-2,0]],[[1,1,1,0]]])
# t_kf = [1]

trajectory = Trajectory(traj, np.cumsum(t_kf))

init_pos, init_control = q.setStateAtNomTime(trajectory, 0)

total_steps = int(np.floor((np.cumsum(t_kf))[-1]/ dt))

print(init_pos, init_control)
est_state = np.zeros((12, total_steps))
est_state[:,0] = init_pos
print((est_state).shape)
u = np.zeros((4,total_steps-1))
print('-------- Executing the path --------')

# connect to the AirSim simulator
# client = airsim.MultirotorClient()
# client.confirmConnection()
# client.enableApiControl(True)
# client.armDisarm(True)

# Take off and move to an initial starting point
# airsim.wait_key('Press any key to takeoff')
# client.takeoffAsync().join()
# client.moveToPositionAsync(init_pos[0], init_pos[1], init_pos[2], 5).join()
# airsim.wait_key('Now we start our task!')

for i in range(total_steps-1):
    # u = q.PDController(trajectory, airsim = True)
    # u[0] = get_throttle(u[0], throttle_const, g)
    ## bound_control(u, max_abs_roll_rate, max_abs_pitch_rate, max_abs_yaw_rate)
    # rotmat_u = np.array([[1,0,0],[0,-1,0],[0,0,-1]])
    # u[1:4] = rotmat_u @ u[1:4]
    # client.moveByAngleRatesThrottleAsync(u[1], u[2], u[3], u[0], dt).join()
    # x = get_drone_state((client.getMultirotorState()).kinematics_estimated, 12)
    # q.setState(x)
    # q.setSystemTime(i * dt)

    ui = q.PDController(trajectory)
    est_state[:,i+1],_ = q.EulStepNlDyn(ui)
    u[:,i] = ui.copy()

# airsim.wait_key('Phew!')
# client.armDisarm(False)
# client.reset()

# client.enableApiControl(False)

print('-------- Done Executing --------')

#############################
# ########### plotting paths

#### plot the trajectory
traj_ptx = []
traj_pty = []
traj_ptz = []
traj_ptp = []
# dt = 0.01
for num in range(len(t_kf)):
    # if num == 0:
    #     T = t_kf[num]
    # else: 
    #     T = t_kf[num]# - t_kf[num-1]
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

plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(traj_ptx, traj_pty, traj_ptz, 'gray')
ax.scatter3D(wpts_test[:,0],wpts_test[:,1],wpts_test[:,2], color = 'g')
ax.scatter3D(
    est_state[0,:], est_state[1,:], 
    est_state[2,:], color = 'b')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
# ax.set_xlim(0, 10)
# ax.set_ylim(0, 10)
ax.set_zlim(-80, 0)
plt.show()



