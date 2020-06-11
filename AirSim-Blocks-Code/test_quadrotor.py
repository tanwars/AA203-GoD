'''
Just a tester script - a sandbox. 
Is not important for the functioning of the package.
'''

from quadrotor import Quadrotor, Trajectory
from mpl_toolkits import mplot3d

import numpy as np
import matplotlib.pyplot as plt

dt = 0.01
q = Quadrotor(dt = dt)

n = 7
m = 1

# weird motion
# traj = np.array([[[0,0,0,0]],[[0,0,0,0]],[[0,1,0,0]],
#                 [[1,0,0,0]],[[1,1,1,1]],[[-2,-2,-2,-2]],[[1,1,1,1]]])
# t_kf = [1]

# no yaw
# x^3  + x^2 - 2x + 1
# y^4 + y^2 -2y + 1
# z^2 - 2z + 1
traj = np.array([[[0,0,0,0]],[[0,0,0,0]],[[0,1,0,0]],
                [[1,0,0,0]],[[1,1,1,0]],[[-2,-2,-2,0]],[[1,1,1,0]]])
t_kf = [1]

# random traj
# temp = np.random.rand(7*1*4)
# sigma_half = 10 * np.eye(28)
# traj = np.reshape(sigma_half @ temp, (7,1,4))
# t_kf = [1]

# no yaw, quadratic x and linear y,z
# traj = np.array([[[0,0,0,0]],[[0,0,0,0]],[[0,0,0,0]],
#                 [[0,0,0,0]],[[1,0,0,0]],[[-2,1,1,0]],[[1,1,1,0]]])
# t_kf = [1]

# yaw, quadratic x, linear y,z
# traj = np.array([[[0,0,0,0]],[[0,0,0,0]],[[0,0,0,0]],
#                 [[0,0,0,0]],[[1,0,0,1]],[[-2,1,1,1]],[[1,1,1,0]]])
# t_kf = [1]

# yaw, quadratic x and linear y,z
# traj = np.array([[[0,0,0,0]],[[0,0,0,0]],[[0,0,0,0]],
#                 [[0,0,0,0]],[[1,0,0,0]],[[-2,1,1,1]],[[1,1,1,0]]])
# t_kf = [1]

# vertical motion
# traj = np.array([[[0,0,0,0]],[[0,0,0,0]],[[0,0,0,0]],
#                 [[0,0,0,0]],[[0,0,0,0]],[[0,0,1,0]],[[1,1,1,0]]])
# t_kf = [1]

# linear motion
# traj = np.array([[[0,0,0,0]],[[0,0,0,0]],[[0,0,0,0]],
#                 [[1,0,0,0]],[[1,1,1,0]],[[1,1,1,0]],[[1,1,1,0]]])
# t_kf = [1]

# linear motion in plane
# traj = np.array([[[0,0,0,0]],[[0,0,0,0]],[[0,0,0,0]],
#                 [[0,0,0,0]],[[0,0,0,0]],[[1,1,0,1]],[[1,1,1,0]]])
# t_kf = [1]

# adam's traj
# traj = np.array([[[-5.3410, 5.6389,0,0],[0.0096,-0.0099,0,0]],
#         [[22.9256, -23.9173,0,0],[-0.0831, 0.0120,0,0 ]],
#         [[-35.9612, 36.5192,0,0],[0.1200, 0.5886 ,0,0]],
#         [[21.3766, -20.2408,0,0],[0.7739, -2.8784,0,0]],
#         [[0,0,0,0],[-2.5284, 3.2461,0,0]],
#         [[0,0,0,0],[0.3707, 3.4055 ,0,0]],
#         [[0,0,0,0],[5.8670, -2.3985,0,0]],
#         [[0,0,0,0],[3.0000, -2.0000,0,0]]])
# t_kf = [1,4]

#### plot the trajectory
traj_ptx = []
traj_pty = []
traj_ptz = []
traj_ptp = []
for num in range(len(t_kf)):
    if num == 0:
        T = t_kf[num]
    else: 
        T = t_kf[num] - t_kf[num-1]
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

##################

trajectory = Trajectory(traj, t_kf)
# x,u = q.diffFlatStatesInputs(trajectory)

#####
# checking the trajectory class
#####
# mytimet = 1
# print(trajectory.sigma(mytimet))
# print(trajectory.sigma(mytimet,1))
# print(trajectory.sigma(mytimet,2))
# print(trajectory.sigma(mytimet,3))
# print(trajectory.sigma(mytimet,4))


print('---------------------------')
print('---- Control Stuff --------')

# init_pos = x[:,0]
# q.setState(init_pos)
# est_state = np.zeros_like(x)
# est_state[:,0] = np.array(init_pos)
init_pos = q.setStateAtNomTime(trajectory, 0)
# init_pos = np.array([ 1., 1., 1., 0., 0.,  0., -2., -2., -2., 0., 0., 0.])
# q.setState(init_pos)
total_steps = int(np.floor((t_kf[-1]/ dt)))
est_state = np.zeros((12, total_steps))
print(init_pos)
est_state[:,0] = init_pos

# feedforward control from diff flat
# for i in range(u.shape[1]-1):
#     est_state[:,i+1],_ = q.EulStepNlDyn(u[:,i])

# # euler step controller
u = np.zeros((4,total_steps-1))
for i in range(u.shape[1]):
    ui = q.PDController(trajectory)
    est_state[:,i+1],_ = q.EulStepNlDyn(ui)
    u[:,i] = ui.copy()

# ode45
# t, est_state = q.solveSystem(init_pos, t_kf[-1], trajectory)

print((est_state).shape)

# print('---------------------------')


##############################################################
# plotting
##############################################################

plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(traj_ptx, traj_pty, traj_ptz, 'gray')
# ax.scatter3D(x[0,:], x[1,:], x[2,:], color = 'g')
ax.scatter3D(
    est_state[0,:], est_state[1,:], 
    est_state[2,:], color = 'b')
# ax.set_xlim(0, 1.5)
# ax.set_ylim(0, 1.5)
# ax.set_zlim(0, 1.5)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

# plt.figure()
# ax = plt.axes(projection='3d')
# ax.scatter3D(x[3,:], x[4,:], x[5,:], color = 'g')
# ax.scatter3D(
#     est_state[3,:], est_state[4,:], 
#     est_state[5,:], color = 'b')
# ax.set_xlabel('phi')
# ax.set_ylabel('theta')
# ax.set_zlabel('psi')

# plt.figure()
# ax = plt.axes(projection='3d')
# ax.plot3D(x[0,:], x[1,:], x[2,:], 'gray')
# ax.set_xlim(0, 10)
# ax.set_ylim(0, 10)
# ax.set_zlim(0, 10)

plt.figure()
# plt.plot(x[0,:])
# plt.plot(x[1,:])
plt.plot(est_state[4,:] * 180/np.pi)

plt.figure()
plt.plot(u[0,:], label = 'u0')
plt.plot(u[1,:], label = 'u1')
plt.plot(u[2,:], label = 'u2')
plt.plot(u[3,:], label = 'u3')
plt.legend()
plt.show()
