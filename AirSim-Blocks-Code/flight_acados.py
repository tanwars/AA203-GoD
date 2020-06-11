'''
precursor to the flight_acados_minsnap.py file. 
Has basically the same thing, except we are not following a min-snap trajectory
but the set of waypoints 
'''

import airsim

import numpy as np
from scipy.signal import cont2discrete
from scipy import linalg as la
from drone_util import get_drone_state, not_reached, get_throttle, bound_control

from models import linear_quad_model, acados_linear_quad_model

from acados_template import AcadosOcp, AcadosOcpSolver


##################################################
##################################################
# Designing the dLQR controller gain
##################################################

# some parameters
g = 9.81                    # acc. due to gravity
dt = 0.01                    # time step
n = 9                       # state dimension
m = 4                       # control input dimension
kq = 1                      # parameter to tune the Q matrix Q = kq * I 
kr = 30                # parameter to tune the R matrix R = kr * I
wpfile = 'waypts_test.csv'  # waypoint file

Tf = 0.5 # this works
# Tf = 1.0 # doesnt work
dist_check = 15.0           # distance to waypoint to say waypoint reached
throttle_const = 0.59375    # constant mapping u4 to throttle(t): t = tc * u4 
max_abs_roll_rate = 50.0     # clamping roll rate
max_abs_pitch_rate = 50.0    # clamping pitch rate
max_abs_yaw_rate = 50.0      # clamping yaw rate
max_iter = 400              # maximum iterations to time out of the loop
ue = np.array([0,0,0,g])    # nominal control
x0 = np.array([0,0,-20,0,0,0,0,0,0])

# linearize the non-linear quadrotor model
# Ac, Bc = linear_quad_model(num_states = 9, g = g)

# make the acados system model

model = acados_linear_quad_model(9, g = g)

# create ocp object to formulate the OCP
ocp = AcadosOcp()
ocp.model = model

nx = model.x.size()[0]
nu = model.u.size()[0]
ny = nx + nu
ny_e = nx
N = int(Tf/dt)
ocp.dims.N = N

# set cost
Q = kq * np.eye(n)
R = kr * np.eye(m)

ocp.cost.W_e = Q
ocp.cost.W = la.block_diag(Q, R)

ocp.cost.cost_type = 'LINEAR_LS'
ocp.cost.cost_type_e = 'LINEAR_LS'

ocp.cost.Vx = np.zeros((ny, nx))
ocp.cost.Vx[:nx,:nx] = np.eye(nx)

Vu = np.zeros((ny, nu))
Vu[nx:,:] = np.eye(nu)
ocp.cost.Vu = Vu

ocp.cost.Vx_e = np.eye(nx)

# solver options

ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # FULL_CONDENSING_QPOASES
ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
ocp.solver_options.integrator_type = 'ERK'
ocp.solver_options.print_level = 0
ocp.solver_options.nlp_solver_type = 'SQP_RTI' # SQP_RTI, SQP
ocp.solver_options.tf = Tf

# constraints on control
ocp.constraints.lbu = np.array([-max_abs_roll_rate, -max_abs_pitch_rate, -max_abs_pitch_rate,0])
ocp.constraints.ubu = np.array([max_abs_roll_rate, max_abs_pitch_rate, max_abs_pitch_rate,1])
ocp.constraints.idxbu = np.array([0,1,2,3])
ocp.constraints.x0 = x0

ocp.cost.yref  = np.zeros((n+m,))
ocp.cost.yref_e = np.zeros((n,))

ocp_solver = AcadosOcpSolver(ocp, json_file = 'acados_ocp.json')

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


# load waypoints
wpts = np.loadtxt(wpfile)
num_wpts = wpts.shape[0]

# looping for waypoint navigation
pt_reached = -1
curr_wpt_state = np.zeros((n,))
while (pt_reached < num_wpts-1):
    curr_wpt = wpts[pt_reached+1,:]
    curr_wpt_state[0:3] = curr_wpt
    yref  = np.hstack((curr_wpt_state, ue))
    yref_e = curr_wpt_state

    x = get_drone_state((client.getMultirotorState()).kinematics_estimated, n)

    for num1 in range(N):
        ocp_solver.set(num1, "yref", yref)
    ocp_solver.set(N, "yref", yref_e)

    # print('hereout1')
    # ocp_solver = AcadosOcpSolver(ocp, json_file = 'acados_ocp.json')
    # print('hereout2')

    iter_num = 0

    while (not_reached(curr_wpt_state, x, dist_check) and iter_num < max_iter):
        # get control
        status = ocp_solver.solve()
        # print('here%d'%iter_num)
        if status != 0:
            raise Exception('acados returned status {}. Exiting.'.format(status))

        u = ocp_solver.get(0, "u")
        est_x = ocp_solver.get(1, "x")
        # print('est state: %f %f %f'%(est_x[0],est_x[1],est_x[2]))
        # u[3] = get_throttle(u[3], throttle_const, g)
        # bound_control(u, max_abs_roll_rate, max_abs_pitch_rate, max_abs_yaw_rate)
        rotmat_u = np.array([[1,0,0],[0,-1,0],[0,0,-1]])
        u[0:3] = rotmat_u @ u[0:3]
        # print(u)
        client.moveByAngleRatesThrottleAsync(u[0], u[1], u[2], u[3], dt).join()
        x = get_drone_state((client.getMultirotorState()).kinematics_estimated, n)
        # print('drone state: %f %f %f'%(x[0],x[1],x[2]))

        ocp_solver.set(0, "lbx", x)
        ocp_solver.set(0, "ubx", x)

        iter_num += 1

    pt_reached += 1
    if (iter_num == max_iter):
        print('max iterations reached; moving to next waypoint')
    else:
        print('Reached waypoint %d' % pt_reached)
    # del ocp_solver    

airsim.wait_key('Phew!')
client.armDisarm(False)
client.reset()

# that's enough fun for now. let's quit cleanly
client.enableApiControl(False)
