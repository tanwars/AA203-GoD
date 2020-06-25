import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def plot_traj(traj, t_kf, dt):
    # calculate the nominal path
    traj_ptx = []
    traj_pty = []
    traj_ptz = []
    traj_ptp = []
    traj_endpts_yaw = []
    traj_endpts_ptx = []
    traj_endpts_pty = []
    traj_endpts_ptz = []
    end_pts_time = []
    all_time = []
    time_add = 0
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
            all_time.append(t + time_add)
            if i == 0:
                traj_endpts_yaw.append(psi(t))
                traj_endpts_ptx.append(x(t))
                traj_endpts_pty.append(y(t))
                traj_endpts_ptz.append(z(t))
                end_pts_time.append(t + time_add)
            if i == int(T/dt) - 1 and num == len(t_kf) - 1:
                traj_endpts_yaw.append(psi(t))
                traj_endpts_ptx.append(x(t))
                traj_endpts_pty.append(y(t))
                traj_endpts_ptz.append(z(t))
                end_pts_time.append(t + time_add)
        time_add += T
    
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(traj_ptx, traj_pty, traj_ptz, 'gray')
    ax.scatter(traj_endpts_ptx, traj_endpts_pty, traj_endpts_ptz)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.gca().invert_zaxis()
    plt.gca().invert_xaxis()
    # ax.set_zlim(0,-80)

    plt.figure()
    plt.plot(all_time,np.array(traj_ptp) * (180 / np.pi), label = 'nominal yaw')
    plt.scatter(end_pts_time,np.array(traj_endpts_yaw) * 180 / np.pi, marker = 'o', label = 'wpts')
    plt.legend()

    # plt.show()

def plot_single_traj(traj, t_kf, dt):
    # calculate the nominal path
    traj_ptx = []
    traj_endpts = []
    end_pts_time = []
    all_time = []
    time_add = 0
    # dt = 0.01

    for num in range(len(t_kf)):
        T = t_kf[num]
        x = np.poly1d(traj[:,num])
        for i in range(int(T/dt)):
            t = i * dt
            traj_ptx.append(x(t))
            all_time.append(t + time_add)
            if i == 0:
                traj_endpts.append(x(t))
                end_pts_time.append(t + time_add)
            if i == int(T/dt) - 1 and num == len(t_kf) - 1:
                traj_endpts.append(x(t))
                end_pts_time.append(t + time_add)
        time_add += T

    plt.figure()
    plt.plot(all_time,np.array(traj_ptx), label = 'nominal')
    plt.scatter(end_pts_time,np.array(traj_endpts), marker = 'o', color = 'y', label = 'wpts')
    plt.legend()

def plot_control(u):
    plt.figure()
    plt.plot(u[:,0], label = 'roll rate')
    plt.plot(u[:,1], label = 'pitch rate')
    plt.plot(u[:,2], label = 'yaw rate')
    plt.plot(u[:,3], label = 'throttle')
    plt.plot(np.ones((u.shape[0],)), ':', label = 'throttle limit')
    plt.ylim([-5,5])
    plt.legend()

def plot_est_traj(traj, t_kf, dt, all_est_path, show_est_path = False):
    # calculate the nominal path
    traj_ptx = []
    traj_pty = []
    traj_ptz = []
    traj_ptp = []
    traj_endpts_yaw = []
    traj_endpts_ptx = []
    traj_endpts_pty = []
    traj_endpts_ptz = []
    end_pts_time = []
    all_time = []
    time_add = 0
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
            all_time.append(t + time_add)
            if i == 0:
                traj_endpts_yaw.append(psi(t))
                traj_endpts_ptx.append(x(t))
                traj_endpts_pty.append(y(t))
                traj_endpts_ptz.append(z(t))
                end_pts_time.append(t + time_add)
            if i == int(T/dt) - 1 and num == len(t_kf) - 1:
                traj_endpts_yaw.append(psi(t))
                traj_endpts_ptx.append(x(t))
                traj_endpts_pty.append(y(t))
                traj_endpts_ptz.append(z(t))
                end_pts_time.append(t + time_add)
        time_add += T
    
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(traj_ptx, traj_pty, traj_ptz, 'gray')
    ax.scatter(traj_endpts_ptx, traj_endpts_pty, traj_endpts_ptz)
    if show_est_path:
        ax.scatter3D(
            all_est_path[:,0], all_est_path[:,1], 
            all_est_path[:,2], color = 'b')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.gca().invert_zaxis()
    plt.gca().invert_xaxis()
    # ax.set_zlim(0,-80)

    plt.figure()
    plt.plot(all_time,np.array(traj_ptp) * (180 / np.pi), label = 'nominal yaw')
    plt.scatter(end_pts_time,np.array(traj_endpts_yaw) * 180 / np.pi, marker = 'o', label = 'wpts')
    plt.legend()
