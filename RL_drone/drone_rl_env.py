import sys
from baseline_racer import BaselineRacer
from utils import to_airsim_vector, to_airsim_vectors
import airsimneurips as airsim
import argparse
import numpy as np
import time
import datetime
import math
import threading
import os
import random
import matplotlib


class BaselineRacerEnv(BaselineRacer):
    def __init__(self, drone_name, drone_params,use_vel_constraints=False,race_tier=1,viz_traj=True):
        super().__init__(drone_name=drone_name)
        self.drone_name = drone_name
        self.drone_params = drone_params
        self.use_vel_constraints = use_vel_constraints
        self.viz_traj = viz_traj

        self.gates_complete = -1
        self.logPath = "C:/Users/shubh/Documents/AirSim/AirSimExe/Saved/Logs/RaceLogs"

        self.race_tier = race_tier

        self.action_high = 1
        self.action_low = -1
        self.max_vel = 40

        self.observation_space = [33,None]
        self.action_space = [4,None]
        
        self.reward = 0
        self.prev_gate = -1
        
        drone_state  = self.airsim_client.getMultirotorState(self.drone_name)
        self.time = drone_state.timestamp
        self.prev_time = drone_state.timestamp
        self.last_gate_change_time = drone_state.timestamp
        start_position = drone_state.kinematics_estimated.position
        self.cur_waypt = airsim.Vector3r(start_position.x_val, start_position.y_val, start_position.z_val-1.0)

        self.disq = False
        self.scene_ls = self.airsim_client.simListSceneObjects()
        self.scene_poses = []
        for obj in self.scene_ls:
            self.scene_poses.append(self.airsim_client.simGetObjectPose(obj).position.to_numpy_array())
        self.scene_poses = np.array(self.scene_poses)

    def reset_race(self):
        self.airsim_client.simResetRace()
        
        time.sleep(1)
        self.gates_complete = -1
        self.reward = 0
        self.disq = False
        self.prev_gate = -1
        drone_state  = self.airsim_client.getMultirotorState(self.drone_name)
        self.time = drone_state.timestamp
        self.prev_time = drone_state.timestamp
        self.last_gate_change_time = drone_state.timestamp
        start_position = drone_state.kinematics_estimated.position
        self.cur_waypt = airsim.Vector3r(start_position.x_val, start_position.y_val, start_position.z_val-1.0)

        self.start_race(self.race_tier)
        self.takeoff_with_moveOnSpline()
        time.sleep(1)

        return self.state_to_feature()

    def state_to_feature(self):
        drone_state  = self.airsim_client.getMultirotorState(self.drone_name)
        position = drone_state.kinematics_estimated.position.to_numpy_array()
        orientation = drone_state.kinematics_estimated.orientation.to_numpy_array()
        # orientation = orientation/ np.linalg.norm(orientation)
        linear_velocity = drone_state.kinematics_estimated.linear_velocity.to_numpy_array()
        # linear_velocity = linear_velocity/ np.linalg.norm(linear_velocity)
        angular_velocity = drone_state.kinematics_estimated.angular_velocity.to_numpy_array()
        # angular_velocity = angular_velocity/ np.linalg.norm(angular_velocity)
        # print(self.gates_complete)
        cur_gate = self.gate_poses_ground_truth[self.gates_complete+1].position.to_numpy_array()
        # cur_gate = np.array([cur_gate.x_val,cur_gate.y_val,cur_gate.z_val])
        
        cur_gate_or = self.gate_poses_ground_truth[self.gates_complete+1].orientation.to_numpy_array()
        # cur_gate_or = cur_gate_or/ np.linalg.norm(cur_gate_or)
        # cur_gate_or = np.array([cur_gate_or.x_val,cur_gate_or.y_val,cur_gate_or.z_val,cur_gate_or.w_val])

        pos_feat = cur_gate - position
        pos_feat_abs = np.linalg.norm(pos_feat)

        height = position[-1]
        # pos_feat = pos_feat/pos_feat_abs 

        scene_dist = np.linalg.norm(self.scene_poses - position[None, :], axis=1)
        scene_idx = np.argsort(scene_dist)[:5]
        scene_feat = (self.scene_poses[scene_idx, :] - position[None, :]).reshape(-1)

        feat = np.r_[orientation,linear_velocity,angular_velocity,pos_feat,height, cur_gate_or, scene_feat]

        # feat = feat / np.linalg.norm(feat)

        return feat

    def calculate_waypoint(self, action = [0, 0, 0, 1.]):
        cur_wp = self.cur_waypt.to_numpy_array()
        cur_gate = self.gate_poses_ground_truth[self.gates_complete+1].position.to_numpy_array()
        position = self.airsim_client.simGetVehiclePose(self.drone_name).position.to_numpy_array()
        dir_vec = cur_gate - position
        dir_vec = dir_vec / np.linalg.norm(dir_vec)
        new_wp = cur_wp + 3*action[3]*action[:3] + 2*(dir_vec)
        # if new_wp[2]>-1.:
        #     new_wp[2] = -1.
        return to_airsim_vector(new_wp)

    def transition(self,action=[0.,0.,0.,0.]):
        
        waypt = self.calculate_waypoint(action)

        # self.airsim_client.moveByVelocityAsync(action[0],action[1],action[2],vehicle_name=self.drone_name)
        self.airsim_client.moveOnSplineAsync([waypt], vel_max=30.0, acc_max=15.0, 
            add_position_constraint=True, add_velocity_constraint=False, add_acceleration_constraint=False, viz_traj=self.viz_traj, viz_traj_color_rgba=self.viz_traj_color_rgba, vehicle_name=self.drone_name)
        time.sleep(0.5)

        next_state = self.state_to_feature()
        # print(next_state)
        
        reward = self.compute_reward()

        if self.disq or self.gates_complete==len(self.gate_poses_ground_truth):
            done = True
        else:
            done = False

        self.cur_waypt = waypt

        return next_state, reward, done, "OK" 


    def compute_reward(self):
        drone_state  = self.airsim_client.getMultirotorState()

        self.time = drone_state.timestamp
        # time_penalty = self.time - self.prev_time
        collision_penalty = 0.
        # print(drone_state.collision)
        if drone_state.collision.has_collided:
            collision_penalty = 3000.
            self.disq = True
        time_penalty = self.time - self.last_gate_change_time

        gates_passed = 0.
        self.gates_complete = self.airsim_client.simGetLastGatePassed(self.drone_name)
        if self.gates_complete > len(self.gate_poses_ground_truth)+10:
            self.gates_complete = -1
        if self.gates_complete > self.prev_gate:
            gates_passed = 1.
            self.prev_gate = self.gates_complete
            self.last_gate_change_time = self.time
        
        tot_reward = gates_passed*10 - collision_penalty - time_penalty/1e10
        # print(tot_reward, time_penalty)
        
        self.prev_time = drone_state.timestamp
        
        return tot_reward  


    def step(self,action,stall = False):
        if self.time - self.last_gate_change_time >= 3000.0*1e7:
            return self.state_to_feature(),0,True,"Hanged"
        time.sleep(2.)
        return self.transition(action)
