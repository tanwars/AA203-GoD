from quadrotor import Quadrotor
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

# no yaw
traj = np.array([[[0,0,0,0]],[[0,0,0,0]],[[0,1,0,0]],
                [[1,0,0,0]],[[1,1,1,0]],[[-2,-2,-2,0]],[[1,1,1,0]]])

# vertical motion
# traj = np.array([[[0,0,0,0]],[[0,0,0,0]],[[0,0,0,0]],
#                 [[0,0,0,0]],[[0,0,0,0]],[[0,0,1,0]],[[1,1,1,0]]])

# linear motion
# traj = np.array([[[0,0,0,0]],[[0,0,0,0]],[[0,0,0,0]],
#                 [[1,0,0,0]],[[1,1,1,0]],[[1,1,1,0]],[[1,1,1,0]]])

# linear motion in plane
# traj = np.array([[[0,0,0,0]],[[0,0,0,0]],[[0,0,0,0]],
#                 [[0,0,0,0]],[[0,0,0,0]],[[1,1,0,1]],[[1,1,1,0]]])

t_kf = [1]

#### plot the trajectory
traj_ptx = []
traj_pty = []
traj_ptz = []
traj_ptp = []
for num in range(len(t_kf)):
    T = t_kf[num]
    for i in range(int(T/dt)):
        t = i * dt
        x = np.poly1d(traj[:,num,0])
        y = np.poly1d(traj[:,num,1])
        z = np.poly1d(traj[:,num,2])
        psi = np.poly1d(traj[:,num,3])
        traj_ptx.append(x(t))
        traj_pty.append(y(t))
        traj_ptz.append(z(t))
        traj_ptp.append(psi(t))

# plt.figure()
# ax = plt.axes(projection='3d')
# ax.plot3D(traj_ptx, traj_pty, traj_ptz, 'gray')
# ax.set_xlim(0, 1.5)
# ax.set_ylim(0, 1.5)
# ax.set_zlim(0, 1.5)
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')

## plt.figure()
## plt.plot(traj_ptz)
## plt.plot(x[1,:])
## plt.plot(x[2,:])

# plt.show()
##################


x,u = q.diffFlatStatesInputs(traj,t_kf)

print('---------------------------')

init_pos = x[:,0]
q.setState(init_pos)
est_state = np.zeros_like(x)
est_state[:,0] = np.array(init_pos)
for i in range((u.shape[1])-1):
    est_state[:,i+1],_ = q.stepNlDyn(u[:,i])


##############################################################
# plotting
##############################################################

plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(traj_ptx, traj_pty, traj_ptz, 'gray')
ax.scatter3D(x[0,:], x[1,:], x[2,:], color = 'g')
ax.scatter3D(
    est_state[0,:], est_state[1,:], 
    est_state[2,:], color = 'b')
# ax.set_xlim(0, 1.5)
# ax.set_ylim(0, 1.5)
# ax.set_zlim(0, 1.5)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(x[3,:], x[4,:], x[5,:], color = 'g')
ax.scatter3D(
    est_state[3,:], est_state[4,:], 
    est_state[5,:], color = 'b')
ax.set_xlabel('phi')
ax.set_ylabel('theta')
ax.set_zlabel('psi')

# plt.figure()
# ax = plt.axes(projection='3d')
# ax.plot3D(x[0,:], x[1,:], x[2,:], 'gray')
# ax.set_xlim(0, 10)
# ax.set_ylim(0, 10)
# ax.set_zlim(0, 10)

# plt.figure()
# plt.plot(x[0,:])
# plt.plot(x[1,:])
# plt.plot(x[2,:])

plt.figure()
plt.plot(u[0,:], label = 'u0')
plt.plot(u[1,:], label = 'u1')
plt.plot(u[2,:], label = 'u2')
plt.plot(u[3,:], label = 'u3')
plt.legend()
plt.show()
