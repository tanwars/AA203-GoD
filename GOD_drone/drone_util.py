import numpy as np
import types

def get_drone_state(kinematics, num_states):
    state = np.zeros((num_states,))
    
    if num_states == 9:
        position = kinematics.position
        state[0] = position.x_val
        state[1] = position.y_val
        state[2] = position.z_val

        orientation = kinematics.orientation
        phi, theta, psi = quaternion_to_eul(orientation)
        state[3] = phi
        state[4] = theta
        state[5] = psi

        vel = kinematics.linear_velocity
        state[6] = vel.x_val
        state[7] = vel.y_val
        state[8] = vel.z_val

    if num_states == 12:
        position = kinematics.position
        state[0] = position.x_val
        state[1] = position.y_val
        state[2] = position.z_val

        orientation = kinematics.orientation
        phi, theta, psi = quaternion_to_eul(orientation)
        state[3] = phi
        state[4] = theta
        state[5] = psi

        vel = kinematics.linear_velocity
        state[6] = vel.x_val
        state[7] = vel.y_val
        state[8] = vel.z_val

        ang_vel = kinematics.angular_velocity
        state[9] = ang_vel.x_val
        state[10] = ang_vel.y_val
        state[11] = ang_vel.z_val

    return state

def not_reached(pt1, pt2, dist):
    if np.linalg.norm(pt1[0:3] - pt2[0:3]) > dist:
        return True
    else:
        return False

def get_throttle(u, tc, g = 9.81):
    # let t = k*u, we have tc = k*g therefore, k = tc/g
    return (tc/g)*u

def bound_control(u, max_abs_roll_rate, max_abs_pitch_rate, max_abs_yaw_rate):
    max_vals = np.array([max_abs_roll_rate,max_abs_pitch_rate,max_abs_yaw_rate,1.0])
    min_vals = np.array([-max_abs_roll_rate,-max_abs_pitch_rate,-max_abs_yaw_rate,0.0])
    return np.maximum(np.minimum(u,max_vals),min_vals)

def quaternion_to_eul(q):
    q0 = q.w_val
    q1 = q.x_val
    q2 = q.y_val
    q3 = q.z_val
    phi = np.arctan2(2*(q0 * q1 + q2 * q3), 1 - 2*((q1)**2 + (q2)**2))
    theta = np.arcsin(2*(q0 * q2 - q3 * q1))
    psi = np.arctan2(2*(q0 * q3 + q1 * q2), 1 - 2 * ((q2)**2 + (q3)**2))
    return phi, theta, psi

def eul_to_rotmat(phi, theta, psi):
    r1 = np.array([ [1,0,0],
                        [0,np.cos(phi),-np.sin(phi)],
                        [0,np.sin(phi),np.cos(phi)]])
    r2 = np.array([ [np.cos(theta),0,np.sin(theta)],
                    [0,1,0],
                    [-np.sin(theta),0,np.cos(theta)]])               
    r3 = np.array([ [np.cos(psi),-np.sin(psi),0],
                        [np.sin(psi),np.cos(psi),0],
                        [0,0,1]])
    
    return r3 @ r1 @ r2

def quaternion_to_rotmat(q):
    phi, theta, psi = quaternion_to_eul(q)
    return eul_to_rotmat(phi, theta, psi)

def regularize_angle(ref, ang):
    # ang_conv = ang
    tol = -10 * np.pi / 180
    if ref > ang:
        while (ref - ang) > np.pi + tol:
            ang += 2*np.pi
        # ang_conv = ang
    else:
        while (ang - ref) > np.pi + tol:
            ang -= 2*np.pi
        # ang_conv = ang

    # return ang_conv
    return ang

def insert_wpt(wpt_arr, time_arr, psi_arr, wpt1, wpt2):

    pt_to_add = (wpt_arr[wpt1,:] + wpt_arr[wpt2,:])/2
    time_to_add = (time_arr[wpt1])/2
    psi_to_add = (psi_arr[wpt1] + psi_arr[wpt2])/2
    
    time_arr[wpt1] = time_to_add
    time_arr = np.insert(time_arr, wpt2, time_to_add)

    wpt_arr = np.insert(wpt_arr, wpt2, pt_to_add, axis = 0)
    psi_arr = np.insert(psi_arr, wpt2, psi_to_add, axis = 0)
    
    return wpt_arr, time_arr, psi_arr

def shift_wpt(wpt_arr, R, shift_vec, wpt_idx):
    wpt_arr[wpt_idx,:] += R @ shift_vec
    return wpt_arr

def remove_wpt(wpt_arr, time_arr, psi_arr, vel_arr, psiv_arr, vel_idx, 
                                        acc_arr, psia_arr, acc_idx, wpt_idx):
    wpt_arr = np.delete(wpt_arr, wpt_idx, axis = 0)
    psi_arr = np.delete(psi_arr, wpt_idx, axis = 0)
    if wpt_idx == 0:
        time_from_wpt = time_arr[wpt_idx]
        time_arr = np.delete(time_arr, wpt_idx)
    elif wpt_idx == len(time_arr):
        time_to_wpt = time_arr[wpt_idx - 1]
        time_arr = np.delete(time_arr, wpt_idx-1)
    else:
        time_to_wpt = time_arr[wpt_idx - 1]
        time_from_wpt = time_arr[wpt_idx]
        tot_time = time_from_wpt + time_to_wpt
        time_arr[wpt_idx] = tot_time
        time_arr = np.delete(time_arr, wpt_idx-1)

    if vel_idx is not None:
        rem_flag = False
        for i in range(len(vel_idx)):
            if vel_idx[i] == wpt_idx:
                idx_to_remove = i
                rem_flag = True

        if rem_flag:
            vel_arr = np.delete(vel_arr, idx_to_remove, axis = 0)
            psiv_arr = np.delete(psiv_arr, idx_to_remove, axis = 0)
            vel_idx = np.delete(vel_idx, idx_to_remove)
    
    if acc_idx is not None:
        rem_flag = False
        for i in range(len(acc_idx)):
            if acc_idx[i] == wpt_idx:
                idx_to_remove = i
                rem_flag = True

        if rem_flag:
            acc_arr = np.delete(acc_arr, idx_to_remove, axis = 0)
            psia_arr = np.delete(psia_arr, idx_to_remove, axis = 0)
            acc_idx = np.delete(acc_idx, idx_to_remove)

    return wpt_arr, time_arr, psi_arr, vel_arr, psiv_arr, vel_idx, acc_arr, \
                psia_arr, acc_idx  
