from argparse import ArgumentParser
import airsimneurips as airsim
import cv2
import threading
import time
import utils
import numpy as np
import math

# Params
level_name = "Final_Tier_1"
tier=1
drone_name = "drone_1"
takeoff_height = 1.0
viz_traj = True
viz_traj_color_rgba = [1.0, 0.0, 0.0, 1.0]

# Setup
client = airsim.MultirotorClient()
client.confirmConnection()
client.simLoadLevel(level_name)
client.confirmConnection()
time.sleep(2.0)

# Arm drone
client.enableApiControl(vehicle_name=drone_name)
client.arm(vehicle_name=drone_name)

# Set default values for trajectory tracker gains 
traj_tracker_gains = airsim.TrajectoryTrackerGains(kp_cross_track = 5.0, kd_cross_track = 0.0, 
                                                    kp_vel_cross_track = 3.0, kd_vel_cross_track = 0.0, 
                                                    kp_along_track = 0.4, kd_along_track = 0.0, 
                                                    kp_vel_along_track = 0.04, kd_vel_along_track = 0.0, 
                                                    kp_z_track = 2.0, kd_z_track = 0.0, 
                                                    kp_vel_z = 0.4, kd_vel_z = 0.0, 
                                                    kp_yaw = 3.0, kd_yaw = 0.1)
client.setTrajectoryTrackerGains(traj_tracker_gains, vehicle_name=drone_name)
time.sleep(0.2)

# Start race
client.simStartRace(tier=tier)

# Take off
# client.takeoffAsync().join()
# client.reset()
start_position = client.simGetVehiclePose(vehicle_name=drone_name).position
# # print(start_position)
takeoff_waypoint = airsim.Vector3r(start_position.x_val, start_position.y_val, start_position.z_val-takeoff_height)

client.moveOnSplineAsync([takeoff_waypoint], vel_max=15.0, acc_max=5.0, add_position_constraint=True, add_velocity_constraint=False, add_acceleration_constraint=False, viz_traj=viz_traj, viz_traj_color_rgba=viz_traj_color_rgba, vehicle_name=drone_name).join()

print(client.simGetLastGatePassed(drone_name))
# Gates
gate_names_sorted_bad = sorted(client.simListSceneObjects("Gate.*"))
gate_indices_bad = [int(gate_name.split('_')[0][4:]) for gate_name in gate_names_sorted_bad]
gate_indices_correct = sorted(range(len(gate_indices_bad)), key=lambda k: gate_indices_bad[k])
gate_names_sorted = [gate_names_sorted_bad[gate_idx] for gate_idx in gate_indices_correct]

next_pose = client.simGetObjectPose(gate_names_sorted[0])
client.moveOnSplineAsync([next_pose.position], vel_max=15.0, acc_max=5.0, add_position_constraint=True, add_velocity_constraint=False, add_acceleration_constraint=False, viz_traj=viz_traj, viz_traj_color_rgba=viz_traj_color_rgba, vehicle_name=drone_name).join()

print(client.simGetLastGatePassed(drone_name))

# print(client.client.call('simGetObjectScale', "Gate00"))