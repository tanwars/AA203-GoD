import math
import pickle
import sys
import termios
import threading
import time
import tty
from argparse import ArgumentParser

import airsimneurips as airsim
import cv2
import matplotlib.pyplot as plt
import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver
from mpl_toolkits import mplot3d

import utils
from acados_ocp_problem_form import (create_ocp_solver,
                                     create_ocp_solver_separate_gains)
from drone_util import (get_drone_state, insert_wpt, quaternion_to_eul,
                        regularize_angle, remove_wpt, shift_wpt)
from minsnap_trajgen import minSnapTG as mstg
from minsnap_trajgen import minSnapTG_without_opt as mstgwoopt
from minsnap_trajgen import minSnapTG_without_opt_new as mstgwoopt_new
from models import (acados_linear_quad_model,
                    acados_linear_quad_model_moving_eq,
                    acados_nonlinear_quad_model, linear_quad_model)
from plot_utils import plot_control, plot_est_traj, plot_traj
from quadrotor import Quadrotor, Trajectory

class MyRacer(object):
    def __init__(self, drone_name = "drone_1", viz_traj=True, viz_traj_color_rgba=[1.0, 0.0, 0.0, 1.0], viz_image_cv2=True):
        self.drone_name = drone_name
        self.gate_poses_ground_truth = None
        self.viz_image_cv2 = viz_image_cv2
        self.viz_traj = viz_traj
        self.viz_traj_color_rgba = viz_traj_color_rgba

        self.airsim_client = airsim.MultirotorClient()
        self.airsim_client.confirmConnection()
        # we need two airsim MultirotorClient objects because the comm lib we use (rpclib) is not thread safe
        # so we poll images in a thread using one airsim MultirotorClient object
        # and use another airsim MultirotorClient for querying state commands 
        self.airsim_client_images = airsim.MultirotorClient()
        self.airsim_client_images.confirmConnection()
        self.airsim_client_odom = airsim.MultirotorClient()
        self.airsim_client_odom.confirmConnection()
        self.level_name = None

        self.image_callback_thread = threading.Thread(target=self.repeat_timer_image_callback, args=(self.image_callback, 0.03))
        self.odometry_callback_thread = threading.Thread(target=self.repeat_timer_odometry_callback, args=(self.odometry_callback, 0.02))
        self.is_image_thread_active = False
        self.is_odometry_thread_active = False

        self.plot_callback_thread = threading.Thread(target=self.repeat_timer_plot_callback, args=(self.plot_callback, 0.01))
        self.is_plot_thread_active = False
        self.breakNow = False

        self.MAX_NUMBER_OF_GETOBJECTPOSE_TRIALS = 10 # see https://github.com/microsoft/AirSim-NeurIPS2019-Drone-Racing/issues/38

        self.drone_state = None

    # loads desired level
    def load_level(self, level_name, sleep_sec = 2.0):
        self.level_name = level_name
        self.airsim_client.simLoadLevel(self.level_name)
        self.airsim_client.confirmConnection() # failsafe
        time.sleep(sleep_sec) # let the environment load completely

    # Starts an instance of a race in your given level, if valid
    def start_race(self, tier=3):
        self.airsim_client.simStartRace(tier)

    # arms drone, enable APIs, set default traj tracker gains
    def initialize_drone(self):
        self.airsim_client.enableApiControl(vehicle_name=self.drone_name)
        self.airsim_client.arm(vehicle_name=self.drone_name)

        # set default values for trajectory tracker gains 
        # traj_tracker_gains = airsim.TrajectoryTrackerGains(kp_cross_track = 5.0, kd_cross_track = 0.0, 
        #                                                     kp_vel_cross_track = 3.0, kd_vel_cross_track = 0.0, 
        #                                                     kp_along_track = 0.4, kd_along_track = 0.0, 
        #                                                     kp_vel_along_track = 0.04, kd_vel_along_track = 0.0, 
        #                                                     kp_z_track = 2.0, kd_z_track = 0.0, 
        #                                                     kp_vel_z = 0.4, kd_vel_z = 0.0, 
        #                                                     kp_yaw = 3.0, kd_yaw = 0.1)

        # self.airsim_client.setTrajectoryTrackerGains(traj_tracker_gains, vehicle_name=self.drone_name)
        
        # (kp_roll, ki_roll, kd_roll) = (0.25,0.0,0.0)
        # (kp_pitch, ki_pitch, kd_pitch) = (0.25,0.0,0.0)
        # (kp_yaw, ki_yaw, kd_yaw) = (0.25,0,0)
        # roll_rate_gains = airsim.PIDGains(kp_roll, ki_roll, kd_roll)
        # pitch_rate_gains = airsim.PIDGains(kp_pitch, ki_pitch, kd_pitch)
        # yaw_rate_gains = airsim.PIDGains(kp_yaw, ki_yaw, kd_yaw)
        # angle_rate_gains = airsim.AngleRateControllerGains(roll_rate_gains, pitch_rate_gains,
        #                                         yaw_rate_gains)
        # self.airsim_client.setAngleRateControllerGains(angle_rate_gains)
        # print('------------ Set manual gains')
        time.sleep(0.2)

    def takeoffAsync(self):
        self.airsim_client.takeoffAsync().join()

    # like takeoffAsync(), but with moveOnSpline()
    def takeoff_with_moveOnSpline(self, takeoff_height = 1.0):
        start_position = self.airsim_client.simGetVehiclePose(vehicle_name=self.drone_name).position
        takeoff_waypoint = airsim.Vector3r(start_position.x_val, start_position.y_val, start_position.z_val-takeoff_height)
        # takeoff_waypoint = self.gate_poses_ground_truth[0].position

        self.airsim_client.moveOnSplineAsync([takeoff_waypoint], vel_max=15.0, acc_max=5.0, add_position_constraint=True, add_velocity_constraint=False, 
            add_acceleration_constraint=False, viz_traj=self.viz_traj, viz_traj_color_rgba=self.viz_traj_color_rgba, vehicle_name=self.drone_name).join()

    # stores gate ground truth poses as a list of airsim.Pose() objects in self.gate_poses_ground_truth
    def get_ground_truth_gate_poses(self):
        gate_names_sorted_bad = sorted(self.airsim_client.simListSceneObjects("Gate.*"))
        # gate_names_sorted_bad is of the form `GateN_GARBAGE`. for example:
        # ['Gate0', 'Gate10_21', 'Gate11_23', 'Gate1_3', 'Gate2_5', 'Gate3_7', 'Gate4_9', 'Gate5_11', 'Gate6_13', 'Gate7_15', 'Gate8_17', 'Gate9_19']
        # we sort them by their ibdex of occurence along the race track(N), and ignore the unreal garbage number after the underscore(GARBAGE)
        gate_indices_bad = [int(gate_name.split('_')[0][4:]) for gate_name in gate_names_sorted_bad]
        gate_indices_correct = sorted(range(len(gate_indices_bad)), key=lambda k: gate_indices_bad[k])
        gate_names_sorted = [gate_names_sorted_bad[gate_idx] for gate_idx in gate_indices_correct]
        self.gate_poses_ground_truth = []
        for gate_name in gate_names_sorted:
            curr_pose = self.airsim_client.simGetObjectPose(gate_name)
            counter = 0
            while (math.isnan(curr_pose.position.x_val) or math.isnan(curr_pose.position.y_val) or math.isnan(curr_pose.position.z_val)) and (counter < self.MAX_NUMBER_OF_GETOBJECTPOSE_TRIALS):
                print(f"DEBUG: {gate_name} position is nan, retrying...")
                counter += 1
                curr_pose = self.airsim_client.simGetObjectPose(gate_name)
            assert not math.isnan(curr_pose.position.x_val), f"ERROR: {gate_name} curr_pose.position.x_val is still {curr_pose.position.x_val} after {counter} trials"
            assert not math.isnan(curr_pose.position.y_val), f"ERROR: {gate_name} curr_pose.position.y_val is still {curr_pose.position.y_val} after {counter} trials"
            assert not math.isnan(curr_pose.position.z_val), f"ERROR: {gate_name} curr_pose.position.z_val is still {curr_pose.position.z_val} after {counter} trials"
            self.gate_poses_ground_truth.append(curr_pose)
    
    def get_gate_facing_vector_from_quaternion(self, airsim_quat, scale = 1.0):
        import numpy as np
        # convert gate quaternion to rotation matrix. 
        # ref: https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion; https://www.lfd.uci.edu/~gohlke/code/transformations.py.html
        q = np.array([airsim_quat.w_val, airsim_quat.x_val, airsim_quat.y_val, airsim_quat.z_val], dtype=np.float64)
        n = np.dot(q, q)
        if n < np.finfo(float).eps:
            return airsim.Vector3r(0.0, 1.0, 0.0)
        q *= np.sqrt(2.0 / n)
        q = np.outer(q, q)
        rotation_matrix = np.array([[1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0]],
                                    [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0]],
                                    [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2]]])
        gate_facing_vector = rotation_matrix[:,1]
        # return airsim.Vector3r(scale * gate_facing_vector[0], scale * gate_facing_vector[1], scale * gate_facing_vector[2])
        return gate_facing_vector * scale, rotation_matrix

    # TO DO - convert the gate poses to trajectory object
    def plan_path(self):
        minimizer = 4 # 4 for snap, 3 for jerk
        degree = 2 * minimizer
        time_dilation = 3.5

        # total_time = 100  # final tier 1
        # scaler = 0.7 # final tier 1

        total_time = 25  # qualifier tier 1
        scaler = 0.7 # qualifier tier 1

        # total_time = 5 # soccer_field_easy  
        # scaler = 0.7  # soccer_field_easy

        # total_time = 30  # soccer_field_medium
        # scaler = 0.7 # soccer_field_medium
        
        vel = 10

        x_offset = 0.0
        y_offset = 0.0
        z_offset = 0.0

        wpts_poses = self.gate_poses_ground_truth
        wpts = np.zeros((len(wpts_poses)+1,3))
        start_position = self.airsim_client.simGetVehiclePose(vehicle_name=self.drone_name).position
        start_orientation = self.airsim_client.simGetVehiclePose(vehicle_name=self.drone_name).orientation
        # print(start_position)
        wpts[0,:] = np.array([start_position.x_val, start_position.y_val, start_position.z_val])

        psis = np.zeros((len(wpts_poses)+1,))
        psi_true = np.zeros((len(wpts_poses)+1,))
        psis[0] = (quaternion_to_eul(start_orientation)[2])
        curr_yaw = psis[0]
        psi_true[0] = curr_yaw
        for i, pose in enumerate(wpts_poses):
            # print(i)
            wpts[i+1,0] = pose.position.x_val + x_offset
            wpts[i+1,1] = pose.position.y_val + y_offset
            wpts[i+1,2] = pose.position.z_val + z_offset

            gate_yaw = quaternion_to_eul(pose.orientation)[2] + np.pi/2
            psi_true[i+1] = gate_yaw
            psis[i+1] = regularize_angle(curr_yaw, gate_yaw - np.pi/2) + np.pi/2
            curr_yaw = psis[i+1]

        num_wpts = (wpts.shape)[0]
        wpts_test = wpts.copy()
        psi_test = psis.copy()

        # psi_test = np.zeros((len(wpts_poses)+1,))
        psi_test = np.ones((num_wpts,)) * (quaternion_to_eul(start_orientation)[2])

        ###
        #  find velocity through gates
        
        wpts_vel = np.zeros((len(wpts_poses)+1,3))
        rot_mat_wpts = np.zeros((len(wpts_poses)+1,3,3))
        for i, pose in enumerate(wpts_poses):
            # print(i)
            wpts_vel[i+1,:], rot_mat_wpts[i+1,:,:] = self.get_gate_facing_vector_from_quaternion(pose.orientation, vel) 
        psi_vel = np.zeros_like(psi_test)
        wpts_vel_idx = np.arange(wpts_vel.shape[0])

        # test1 =  1 * (wpts_test[1,:] - wpts_test[0,:])/np.linalg.norm(wpts_test[1,:] - wpts_test[0,:])
        # wpts_vel = np.reshape(test1, (1,3))
        # wpts_vel = np.array([[0,0,-2]])

        wpts_vel = np.array([[0,0,0]])
        
        psi_vel = np.array([0])
        wpts_vel_idx = np.array([0])

        # wpts_acc = np.zeros((len(wpts_poses)+1,3))
        # psi_acc = np.zeros_like(psi_test)
        # wpts_acc_idx = np.arange(wpts_acc.shape[0])
        wpts_acc = None
        psi_acc = None
        wpts_acc_idx = None

        ###
        #  find times between gates

        # find distance between gates
        dist_bw_gates = np.zeros((num_wpts-1,))
        for i in range(num_wpts-1):
            dist_bw_gates[i] = np.linalg.norm(wpts_test[i+1,:] - wpts_test[i,:])
        # normalize the distance
        dist_bw_gates /= np.sum(dist_bw_gates)
        # weight each time segment by the distance, use a total time measure
        t_kf = dist_bw_gates * total_time
        t_kf = np.maximum(t_kf, scaler*np.ones(t_kf.shape))
        
        # t_kf[0] = 3.0
        # # wpts_test, t_kf, psi_test = remove_wpt(wpts_test, t_kf, psi_test, 21)
        # wpts_test, t_kf, psi_test, wpts_vel, psi_vel, wpts_vel_idx, wpts_acc, psi_acc, wpts_acc_idx = remove_wpt(
        #                     wpts_test, t_kf, psi_test, 
        #                     wpts_vel, psi_vel, wpts_vel_idx, 
        #                     wpts_acc, psi_acc, wpts_acc_idx,
        #                     21)

        # wpts_test = shift_wpt(wpts_test,  rot_mat_wpts[4,:,:], np.array([0,3,0]), 4)
        # wpts_test = shift_wpt(wpts_test,  rot_mat_wpts[5,:,:], np.array([0,3,0]), 5)
        # wpts_test = shift_wpt(wpts_test,  rot_mat_wpts[6,:,:], np.array([0,3,0]), 6)

        ## 
        # qualifier tier 1
        t_kf[0] = 1.5
        t_kf[12] = 3.5
        wpts_test = shift_wpt(wpts_test,  rot_mat_wpts[13,:,:], np.array([0,0,-1]), 13)
        wpts_test = shift_wpt(wpts_test,  rot_mat_wpts[1,:,:], np.array([1,0,-1]), 1)
        wpts_test = shift_wpt(wpts_test,  rot_mat_wpts[2,:,:], np.array([0,0,-1]), 2)
        wpts_test = shift_wpt(wpts_test,  rot_mat_wpts[3,:,:], np.array([0,0,-1]), 3)
        wpts_test = shift_wpt(wpts_test,  rot_mat_wpts[4,:,:], np.array([0,0,-1]), 4)
        wpts_test = shift_wpt(wpts_test,  rot_mat_wpts[5,:,:], np.array([0,0,-1]), 5)
        wpts_test = shift_wpt(wpts_test,  rot_mat_wpts[6,:,:], np.array([1,0,-1]), 6)
        wpts_test = shift_wpt(wpts_test,  rot_mat_wpts[7,:,:], np.array([0,0,-1]), 7)
        wpts_test = shift_wpt(wpts_test,  rot_mat_wpts[8,:,:], np.array([0,0,-1]), 8)
        wpts_test = shift_wpt(wpts_test,  rot_mat_wpts[9,:,:], np.array([0,0,-1]), 9)
        wpts_test = shift_wpt(wpts_test,  rot_mat_wpts[14,:,:], np.array([0,0,-1]), 14)
        wpts_test = shift_wpt(wpts_test,  rot_mat_wpts[15,:,:], np.array([0,0,-1]), 15)
        wpts_test = shift_wpt(wpts_test,  rot_mat_wpts[16,:,:], np.array([0,0,-1]), 16)

        ## 
        # for soccer field medium
        # wpts_test, t_kf, psi_test = insert_wpt(wpts_test, t_kf, psi_test, 20, 21)
        # wpts_test, t_kf, psi_test = insert_wpt(wpts_test, t_kf, psi_test, 20, 21)
        # wpts_test, t_kf, psi_test = insert_wpt(wpts_test, t_kf, psi_test, 22, 23)
        # wpts_test = shift_wpt(wpts_test,  rot_mat_wpts[22,:,:], np.array([0,-1,0]), 25)
        # wpts_test = shift_wpt(wpts_test,  rot_mat_wpts[9,:,:], np.array([0,-1,-1]), 9)
        # wpts_test = shift_wpt(wpts_test,  rot_mat_wpts[2,:,:], np.array([0,-1,-1]), 2)
        # wpts_test = shift_wpt(wpts_test,  rot_mat_wpts[3,:,:], np.array([0, 0,-1]), 3)
        # wpts_test = shift_wpt(wpts_test,  rot_mat_wpts[4,:,:], np.array([0, 0,-1]), 4)
        # wpts_test = shift_wpt(wpts_test,  rot_mat_wpts[10,:,:], np.array([0,-1,-2]), 10)
        # wpts_test = shift_wpt(wpts_test,  rot_mat_wpts[12,:,:], np.array([0, 0,-1]), 12)
        # wpts_test = shift_wpt(wpts_test,  rot_mat_wpts[20,:,:], np.array([0,-1,-2]), 20)
        # t_kf[0] = 3.0
        # t_kf[24] = 1.5
        # t_kf[25] = 2.0

        # ### 
        # # for soccer field easy
        # t_kf[0] = 2.0
        # t_kf[1] = 1.0
        # t_kf[2] = 1.0
        
        
        ### in case you want to use time_dilation and constant time between gates
        # t_kf = time_dilation * np.ones((num_wpts-1,))
        # t_kf[0] = time_dilation * 1.5

        print('--------------------')
        print('Waypoints')
        print(wpts_test)
        print('--------------------')

        print('--------------------')
        print('Yaws')
        print(np.stack((psi_true, psis), axis = 1) * 180 / np.pi)
        print('--------------------')

        print('--------------------')
        print('Velocities')
        print(wpts_vel)
        print('--------------------')

        print('--------------------')
        print('Times')
        print(t_kf)
        print('--------------------')
        
        print('--------------------')
        print('There are total %d waypoints. Expected Completion time is %f'%(wpts_test.shape[0], np.cumsum(t_kf)[-1]))
        print('--------------------')

        P = [None] * 4
        for i in range(3):
            # P[i] = mstg(degree, minimizer, t_kf, wpts_test[:,i])
            # P[i] = mstgwoopt(degree, minimizer, t_kf, wpts_test[:,i])
            if wpts_acc_idx is not None:
                P[i] = mstgwoopt_new(degree, minimizer, t_kf, wpts_test[:,i], np.arange(wpts_test.shape[0]), 
                                                    wpts_vel[:,i], wpts_vel_idx,
                                                    wpts_acc[:,i], wpts_acc_idx)
            elif wpts_vel_idx is not None:
                P[i] = mstgwoopt_new(degree, minimizer, t_kf, wpts_test[:,i], np.arange(wpts_test.shape[0]), 
                                                    wpts_vel[:,i], wpts_vel_idx)
            else:
                P[i] = mstgwoopt_new(degree, minimizer, t_kf, wpts_test[:,i], np.arange(wpts_test.shape[0]))
        # P[3] = mstg(degree, minimizer, t_kf, psi_test)
        # P[3] = mstgwoopt(degree, minimizer, t_kf, psi_test)
        P[3] = mstgwoopt_new(degree, minimizer, t_kf, psi_test, np.arange(wpts_test.shape[0]), 
                                                        psi_vel, wpts_vel_idx,
                                                        psi_acc, wpts_acc_idx)
        traj = np.stack(P,axis = 2)

        trajectory = Trajectory(traj, np.cumsum(t_kf))

        # plot_traj(traj, t_kf, 0.1)
        return trajectory, t_kf, traj

    def fly_plan(self, trajectory, t_kf, traj, viz_plots = False):

        # some parameters
        g = 9.81                    # acc. due to gravity
        dt = 0.1                    # time step
        # dt = 0.2
        n = 9                       # state dimension
        m = 4                       # control input dimension
        
        # final gains that work for soccer field medium
        kqxy = 0.1
        kqz = 0.1#10
        kqrp = 0.001#0.1#0.01
        kqy = 0.1
        kqxyv = 0.01
        kqzv = 0.1
        
        krt = 0.0001#0.001
        krrpr = 0.01#0.1
        kryr = 0.1#0.1

        # old ones that work for soccer field medium
        # kqxy = 0.1
        # kqz = 1#10
        # kqrp = 0.001#0.1#0.01
        # kqy = 1
        # kqxyv = 0.01
        # kqzv = 0.1
        
        # krt = 0.0001#0.001
        # krrpr = 0.1#0.1
        # kryr = 0.1#0.1

        # old ones that work for soccer field medium
        # kqxy = 1
        # kqz = 1#10
        # kqrp = 0.01#0.1#0.01
        # kqy = 1
        # kqxyv = 0.01
        # kqzv = 0.1
        
        # krt = 0.0001#0.001
        # krrpr = 0.1#0.1
        # kryr = 0.1#0.1

        # rotate_quad = True  # set to true if you want yaw to change between waypoints
        mode = 'nonlinear'
        v2 = False # set to true if you want to compute jacobian along the way in linear sys

        Tf = 0.5#0.5 # time horizon for MPC
        # Tf = 1.0
        # dist_check = 15.0           # distance to waypoint to say waypoint reached
        # throttle_const = 0.59375    # constant mapping u4 to throttle(t): t = tc * u4 
        max_abs_roll_rate = 5.0#10.0#50.0     # clamping roll rate
        max_abs_pitch_rate = 5.0#10.0#50.0    # clamping pitch rate
        max_abs_yaw_rate = 5.0#10.0#50.0      # clamping yaw rate
        
        # max_abs_roll_rate = 50.0#10.0#50.0     # clamping roll rate
        # max_abs_pitch_rate = 50.0#10.0#50.0    # clamping pitch rate
        # max_abs_yaw_rate = 50.0#10.0#50.0      # clamping yaw rate
        # max_iter = 400              # maximum iterations to time out of the loop
        ue = np.array([0,0,0,g])    # nominal control

        ### 
        # initialize quadrotor model
        q = Quadrotor(dt = dt)
        init_pos, _ = q.setStateAtNomTime(trajectory, 0)
        total_steps = int(np.floor((np.cumsum(t_kf))[-1]/ dt))

        ### 
        # initialize the controller
        if mode == 'nonlinear':
            print('--------------------')
            print('Using Non-linear model')
            print('--------------------')
            model = acados_nonlinear_quad_model(9, g = g)
        else:
            if v2 == True:
                print('--------------------')
                print('Using linear model V2 - same as NL??')
                print('--------------------')
                model = acados_linear_quad_model_moving_eq(9, g = g)
            else:
                print('--------------------')
                print('Using linear model')
                print('--------------------')
                model = acados_linear_quad_model(9, g = g)

        ocp_solver = create_ocp_solver_separate_gains(model, Tf, dt, n, m, 
                            max_abs_roll_rate, max_abs_pitch_rate,
                            max_abs_yaw_rate, init_pos[:n], kqxy, kqz, kqrp, kqy, 
                            kqxyv, kqzv, krt, krrpr, kryr)
        N = int(Tf/dt)

        self.airsim_client.moveToPositionAsync(init_pos[0], init_pos[1], init_pos[2], 5).join()

        prev_yaw = 0
        jump_yaw = np.pi

        ### 
        # logging stuff

        all_controls = np.zeros((total_steps-1,m))
        all_est_path = np.zeros((total_steps-1,n))

        print('-------- Executing the path --------')
        for i in range(total_steps-1):
            
            if self.breakNow == True:
                break

            # q.setSystemTime(i * dt)
            time = i * dt
            for num1 in range(N):
                t = time + num1 * dt
                if num1 == 0:
                    full_state, _ = q.getStateAtNomTime(trajectory, t)
                    # all_diff_flat_states[i,:] = full_state[:n].copy()
                else:
                    full_state, _ = q.getStateAtNomTime(trajectory, t)
                yref  = np.hstack((full_state[:n], ue))
                ocp_solver.set(num1, "yref", yref)
            t = time + N * dt
            full_state, _ = q.getStateAtNomTime(trajectory, t)
            yref_e = full_state[:n].copy()
            ocp_solver.set(N, "yref", yref_e)

            status = ocp_solver.solve()
            if status != 0:
                print('Solver  failed!')
                break
                # raise Exception('acados returned status {}. Exiting.'.format(status))

            u = ocp_solver.get(0, "u")
            est_x = ocp_solver.get(1, "x")

            all_controls[i,:] = u.copy()
            all_est_path[i,:] = est_x.copy()
            # all_diff_flat_states[i,:] = full_state[:n].copy()

            rotmat_u = np.array([[1,0,0],[0,-1,0],[0,0,-1]])
            u[0:3] = rotmat_u @ u[0:3]

            self.airsim_client.moveByAngleRatesThrottleAsync(u[0], u[1], u[2], u[3], dt).join()
            x = get_drone_state(self.drone_state.kinematics_estimated, n)

            # print(u[3])
            # print(x)
            # if (np.abs(prev_yaw - x[5]) > jump_yaw):
            #     x[5] -= 2 * np.pi
            if (prev_yaw - x[5] > jump_yaw):
                x[5] += 2 * np.pi
            elif (prev_yaw - x[5] < -jump_yaw):
                x[5] -= 2 * np.pi
            
            prev_yaw = x[5]
            # all_est_path_asim[i,:] = x.copy()
            ocp_solver.set(0, "lbx", x)
            ocp_solver.set(0, "ubx", x)
        print('-------- Done Executing --------')

        ### 
        # plotting
        if viz_plots:
            plot_control(all_controls)
            plot_est_traj(traj, t_kf, dt, all_est_path, show_est_path = True)
            # plot_est_traj(traj, t_kf, dt, all_est_path)
            plt.show()

    def plot_callback(self):

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)

        try:
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

        if ch == "x": # Or skip this check and just break
            self.breakNow = True

    def image_callback(self):
        # get uncompressed fpv cam image
        request = [airsim.ImageRequest("fpv_cam", airsim.ImageType.Scene, False, False)]
        response = self.airsim_client_images.simGetImages(request)
        img_rgb_1d = np.fromstring(response[0].image_data_uint8, dtype=np.uint8) 
        img_rgb = img_rgb_1d.reshape(response[0].height, response[0].width, 3)
        if self.viz_image_cv2:
            cv2.imshow("img_rgb", img_rgb)
            cv2.waitKey(1)

    def odometry_callback(self):
        # get uncompressed fpv cam image
        drone_state = self.airsim_client_odom.getMultirotorState()
        # in world frame:
        position = drone_state.kinematics_estimated.position 
        orientation = drone_state.kinematics_estimated.orientation
        linear_velocity = drone_state.kinematics_estimated.linear_velocity
        angular_velocity = drone_state.kinematics_estimated.angular_velocity
        self.drone_state = drone_state

    # call task() method every "period" seconds. 
    def repeat_timer_image_callback(self, task, period):
        while self.is_image_thread_active:
            task()
            time.sleep(period)

    def repeat_timer_odometry_callback(self, task, period):
        while self.is_odometry_thread_active:
            task()
            time.sleep(period)

    def repeat_timer_plot_callback(self, task, period):
        while self.is_plot_thread_active:
            task()
            time.sleep(period)

    def start_plot_callback_thread(self):
        if not self.is_plot_thread_active:
            self.is_plot_thread_active = True
            self.plot_callback_thread.start()
            print("Started plotting callback thread")

    def start_image_callback_thread(self):
        if not self.is_image_thread_active:
            self.is_image_thread_active = True
            self.image_callback_thread.start()
            print("Started image callback thread")

    def stop_image_callback_thread(self):
        if self.is_image_thread_active:
            self.is_image_thread_active = False
            self.image_callback_thread.join()
            print("Stopped image callback thread.")

    def start_odometry_callback_thread(self):
        if not self.is_odometry_thread_active:
            self.is_odometry_thread_active = True
            self.odometry_callback_thread.start()
            print("Started odometry callback thread")

    def stop_odometry_callback_thread(self):
        if self.is_odometry_thread_active:
            self.is_odometry_thread_active = False
            self.odometry_callback_thread.join()
            print("Stopped odometry callback thread.")

    def test_throttle_const(self):
        t = 10
        throttle = 0.59375 # works
        # throttle = 0.5875
        print('Testing throttle %f'%throttle)
        self.airsim_client.moveByAngleRatesThrottleAsync(0, 0, 0, min(throttle,1.0), t).join()


def main(args):
    # ensure you have generated the neurips planning settings file by running python generate_settings_file.py
    racer = MyRacer(drone_name="drone_1", viz_traj=args.viz_traj, \
        viz_traj_color_rgba=[1.0, 1.0, 0.0, 1.0], viz_image_cv2=args.viz_image_cv2)
    racer.load_level(args.level_name)
    if args.level_name == "Qualifier_Tier_1":
        args.race_tier = 1
    if args.level_name == "Qualifier_Tier_2":
        args.race_tier = 2
    if args.level_name == "Qualifier_Tier_3":
        args.race_tier = 3
    racer.start_race(args.race_tier)
    racer.initialize_drone()
    racer.get_ground_truth_gate_poses()
    
    # racer.takeoff_with_moveOnSpline(1.0)
    # racer.takeoff_with_moveOnSpline(2.0)
    
    # racer.start_image_callback_thread()
    racer.start_odometry_callback_thread()
    racer.start_plot_callback_thread()

    # racer.test_throttle_const()

    traj, t_kf, traj_o = racer.plan_path()
    racer.fly_plan(traj, t_kf, traj_o, args.viz_plots)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--level_name', type=str, choices=["Soccer_Field_Easy", \
                 "Soccer_Field_Medium", "ZhangJiaJie_Medium", "Building99_Hard", \
        "Qualifier_Tier_1", "Qualifier_Tier_2", "Qualifier_Tier_3", "Final_Tier_1", \
                     "Final_Tier_2", "Final_Tier_3"], default="Soccer_Field_Easy")
    parser.add_argument('--planning_baseline_type', type=str, \
        choices=["all_gates_at_once","all_gates_one_by_one"], default="all_gates_at_once")
    parser.add_argument('--planning_and_control_api', type=str, \
        choices=["moveOnSpline", "moveOnSplineVelConstraints"], default="moveOnSpline")
    parser.add_argument('--enable_viz_traj', dest='viz_traj', action='store_true', default=True)
    parser.add_argument('--enable_plots', dest='viz_plots', action='store_true', default=False)
    parser.add_argument('--enable_viz_image_cv2', dest='viz_image_cv2', action='store_true', default=False)
    parser.add_argument('--race_tier', type=int, choices=[1,2,3], default=1)
    args = parser.parse_args()
    main(args)
