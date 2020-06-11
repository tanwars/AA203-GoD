'''
A simple dLQR controller for waypoint following.
'''

import airsim

import numpy as np
from scipy.signal import cont2discrete
from scipy import linalg as la
from drone_util import get_drone_state, not_reached, get_throttle, bound_control

from models import linear_quad_model

# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# Take off and move to an initial starting point
airsim.wait_key('Press any key to takeoff')
client.takeoffAsync().join()
client.moveToPositionAsync(0, 0, -20, 5).join()
airsim.wait_key('Now we start our task!')

##################################################
##################################################
# Designing the dLQR controller gain
##################################################

# some parameters
g = 9.81                    # acc. due to gravity
dt = 0.1                    # time step
n = 9                       # state dimension
m = 4                       # control input dimension
kq = 1                      # parameter to tune the Q matrix Q = kq * I 
kr = 30000                  # parameter to tune the R matrix R = kr * I
wpfile = 'waypts_test.csv'  # waypoint file

dist_check = 15.0           # distance to waypoint to say waypoint reached
throttle_const = 0.59375    # constant mapping u4 to throttle(t): t = tc * u4 
max_abs_roll_rate = 5.0     # clamping roll rate
max_abs_pitch_rate = 5.0    # clamping pitch rate
max_abs_yaw_rate = 5.0      # clamping yaw rate
max_iter = 400              # maximum iterations to time out of the loop
ue = np.array([0,0,0,g])    # nominal control

# linearize the non-linear quadrotor model
Ac, Bc = linear_quad_model(num_states = 9, g = g)

# discretize the linear quadrotor model
Cc = np.zeros_like(Ac)
Dc = np.zeros_like(Bc)
(Ad,Bd,_,_,_) = cont2discrete((Ac, Bc, Cc, Dc), dt, method='zoh')

# Solve the DARE, and compute the constant gain matrix 
Q = kq * np.eye(n)
R = kr * np.eye(m)
P = la.solve_discrete_are(Ad, Bd, Q, R)
K = (np.linalg.inv(R + np.transpose(Bd) @ P @ Bd)) @ np.transpose(Bd) @ P @ Ad

# load waypoints
wpts = np.loadtxt(wpfile)
num_wpts = wpts.shape[0]

# looping for waypoint navigation
pt_reached = -1
curr_wpt_state = np.zeros((n,))
while (pt_reached < num_wpts-1):
    curr_wpt = wpts[pt_reached+1,:]
    curr_wpt_state[0:3] = curr_wpt
    x = get_drone_state((client.getMultirotorState()).kinematics_estimated, n)
    iter_num = 0

    while (not_reached(curr_wpt_state, x, dist_check) and iter_num < max_iter):
        u = ue - K @ (x - curr_wpt_state)
        u[3] = get_throttle(u[3], throttle_const, g)
        bound_control(u, max_abs_roll_rate, max_abs_pitch_rate, max_abs_yaw_rate)
        rotmat_u = np.array([[1,0,0],[0,-1,0],[0,0,-1]])
        u[0:3] = rotmat_u @ u[0:3]
        client.moveByAngleRatesThrottleAsync(u[0], u[1], u[2], u[3], dt).join()
        x = get_drone_state((client.getMultirotorState()).kinematics_estimated, n)
        iter_num += 1

    pt_reached += 1
    if (iter_num == max_iter):
        print('max iterations reached; moving to next waypoint')
    else:
        print('Reached waypoint %d' % pt_reached)

airsim.wait_key('Phew!')
client.armDisarm(False)
client.reset()

# that's enough fun for now. let's quit cleanly
client.enableApiControl(False)
