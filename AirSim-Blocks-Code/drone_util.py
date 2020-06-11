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