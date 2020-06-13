import os
import sys
import pickle
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from drone_rl_env import *
import time
from td3 import TD3
from tensorboardX import SummaryWriter


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Launch AirSimExe w/o graphics with flag "-nullrhi"
# Kill process with Stop-Process -Name "python/AirSimExe" -Force
# check processes with Get-Process

params = {
    "vel_constraints": False,
    "race_tier": 1,
    "level_name": "Final_Tier_1",
    "rand_steps": 200,
    "explore_steps": 1000,
    "batch_size": 10,
    "gamma": 0.99,
    "tau": 0.005,
    "noise": 0.1,
    "noise_clip": 0.5,
    "explore_noise": 0.1,
    "policy_freq": 2,
    "eval_freq": 5000,
    "reward_thresh": 8000,
    "max_T": 2000
}
 
# src: https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py

# Expects tuples of (state, next_state, action, reward, done)
class ReplayBuffer(object):
    """Buffer to store tuples of experience replay"""
    
    def __init__(self, max_size=1000000):
        """
        Args:
            max_size (int): total amount of tuples to store
        """        
        self.storage = []
        try:
            print("Attempt to load replay buff")
            self.storage = self.load_buff()
        except:
            print("Failed to load replay buff")
            pass
        self.max_size = max_size
        self.ptr = (len(self.storage) + 1) % self.max_size

    def load_buff(self):
        my_file = open("./analysis/replay_buffer","rb")
        return pickle.load(my_file)

    def dump_buff(self):
        my_file = open("./analysis/replay_buffer","wb+")
        pickle.dump(self.storage,my_file)

    def add(self, data):
        """Add experience tuples to buffer
        
        Args:
            data (tuple): experience replay tuple
        """
        
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        """Samples a random amount of experiences from buffer of batch size
        
        Args:
            batch_size (int): size of sample
        """
        
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        states, actions, next_states, rewards, dones = [], [], [], [], []

        for i in ind: 
            s, a, s_, r, d = self.storage[i]
            states.append(np.array(s, copy=False))
            actions.append(np.array(a, copy=False))
            next_states.append(np.array(s_, copy=False))
            rewards.append(np.array(r, copy=False))
            dones.append(np.array(d, copy=False))

        return np.array(states), np.array(actions), np.array(next_states), np.array(rewards).reshape(-1, 1), np.array(dones).reshape(-1, 1)

class Policy:
    def __init__(self, env, mode="random"):
        self.mode = mode
        self.env = env
    
    def select_action(self, state, noise=0.1):
        
        action = np.zeros(4)
        action[:3] = state[10:13]
        action[3] = np.linalg.norm(action[:3])
        action[:3] = action[:3]/action[3]
        action[3] = 1.
        # action = np.random.normal(self.env.action_low, self.env.action_high-self.env.action_low, size=self.env.action_space[0])


        if noise != 0: 
            action = (action + np.random.normal(0, noise, size=self.env.action_space[0]))
            
        return action.clip(self.env.action_low, self.env.action_high)

def initialize_buf(env,replay_buffer, observation_steps, policy):
    time_steps = 0
    obs = env.reset_race()
    done = False

    while time_steps < observation_steps:

        print(time_steps)

        action = policy.select_action(obs)

        # action[0:3] = action[0:3]*2 -1

        new_obs, reward, done, _ = env.step(action)

        replay_buffer.add((obs, new_obs, action, reward, float(done)))

        obs = new_obs
        time_steps += 1

        if done:
            obs = env.reset_race()
            done = False

        print("\rInitializing Replay Buffer {}/{}".format(time_steps, observation_steps))
        sys.stdout.flush()

def eval(agent, env, writer, params):
    """Eval the agent for exploration steps
    
        Args:
            agent (Agent): agent to use
            env (environment): gym environment
            writer (SummaryWriter): tensorboard writer
            exploration (int): how many training steps to run
    """
    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = agent.ep_num
    episode_reward = 0
    episode_timesteps = params["rand_steps"]
    done = False 
    obs = env.reset_race()
    time.sleep(1.0)
    rewards = []
    best_avg = -100000

    agent.eval_mode()
    
    traj_ls = []
    temp_traj = []

    while total_timesteps < params["explore_steps"]:
    
        if done: 

            rewards.append(episode_reward)
            
            avg_reward = np.mean(rewards[-50:])
            
            agent.ep_num = episode_num
            
            print("T: {:d} Episode Num: {:d} Reward: {:f} Avg Reward: {:f}".format(total_timesteps, episode_num, episode_reward, avg_reward), end="\n")

            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1 

            agent.adjust_learning_rate(episode_num)

            obs = env.reset_race()

            traj_ls.append(temp_traj)
            temp_traj = []

            time.sleep(1.0)

        pose = env.airsim_client.simGetVehiclePose(env.drone_name).position.to_numpy_array()
        temp_traj.append(pose)
        action = agent.select_action(np.array(obs), noise=0.1)
        new_obs, reward, done, _ = env.step(action) 
        obs = new_obs
        episode_reward += reward
        if episode_timesteps > params["max_T"]:
            done = True
        episode_timesteps += 1
        total_timesteps += 1
        timesteps_since_eval += 1

        if total_timesteps%100==0:
            print(total_timesteps,"-",episode_num)

    traj_ls.append(temp_traj)
    for i, path in enumerate(traj_ls):
        np.savetxt("traj_1_"+str(i)+".txt", np.array(path))

def init_drone_env(params):
    drone_name = "drone_1"
    drone_params = {"v_max": 80.0,
         "a_max": 40.0}

    drone = BaselineRacerEnv(
        drone_name=drone_name,
        drone_params=drone_params,
        use_vel_constraints=params["vel_constraints"],
        race_tier=params["race_tier"])

    drone.level_name = params["level_name"]
    drone.race_tier = params["race_tier"]
    drone.load_level(params["level_name"])

    return drone

################################################################################
# Exec
env = init_drone_env(params)
env.start_race(params["race_tier"])
env.initialize_drone()
env.takeoff_with_moveOnSpline()
env.get_ground_truth_gate_poses()
print([[en.position.x_val, en.position.y_val, en.position.z_val] for en in env.gate_poses_ground_truth])
raise

state_dim = env.observation_space[0]
action_dim = env.action_space[0] 
max_action = float(env.action_high)

total_timesteps = 0
timesteps_since_eval = 0
episode_num = 0
done = True
writer = SummaryWriter(log_dir="./logs")

policy = TD3(state_dim, action_dim, max_action, env)

eval(policy, env, writer, params)

time.sleep(2.0)