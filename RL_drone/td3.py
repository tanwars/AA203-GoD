import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import os
import sys
import pickle

import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# from baselines.common.segment_tree import SumSegmentTree, MinSegmentTree
def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Initialize parameters and build model.
        Args:
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            max_action (float): highest action to take
            seed (int): Random seed
            h1_units (int): Number of nodes in first hidden layer
            h2_units (int): Number of nodes in second hidden layer
            
        Return:
            action output of network with tanh activation
    """
    
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 32)
        self.bn1 = nn.BatchNorm1d(num_features=32)
        self.l2 = nn.Linear(32, 16)
        self.bn2 = nn.BatchNorm1d(num_features=16)
        self.l3 = nn.Linear(16, action_dim-1)
        self.l4 = nn.Linear(16, 1)

        self.max_action = max_action


    def forward(self, x):
        x = F.relu(self.bn1(self.l1(x)))
        x = F.relu(self.bn2(self.l2(x)))
        vec = torch.tanh(self.l3(x))
        vel = torch.sigmoid(self.l4(x))  
        action = torch.cat([vec,vel],1)
        return action

class Critic(nn.Module):
    """Initialize parameters and build model.
        Args:
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            max_action (float): highest action to take
            seed (int): Random seed
            h1_units (int): Number of nodes in first hidden layer
            h2_units (int): Number of nodes in second hidden layer
            
        Return:
            value output of network 
    """
    
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        # self.l1 = nn.Linear(state_dim + action_dim, 32)
        # self.bn1 = nn.BatchNorm1d(num_features=32)
        # self.l2 = nn.Linear(32, 16)
        # self.bn2 = nn.BatchNorm1d(num_features=16)
        # self.l3 = nn.Linear(16, 1)

        self.l1 = nn.Linear(state_dim + action_dim, 32)
        self.bn1 = nn.BatchNorm1d(num_features=32)
        self.l2 = nn.Linear(32, 16)
        self.bn2 = nn.BatchNorm1d(num_features=16)
        self.l3 = nn.Linear(16, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 32)
        self.bn3 = nn.BatchNorm1d(num_features=32)
        self.l5 = nn.Linear(32, 16)
        self.bn4 = nn.BatchNorm1d(num_features=16)
        self.l6 = nn.Linear(16, 1)


    def forward(self, x, u):
        xu = torch.cat([x, u], 1)

        x1 = F.relu(self.bn1(self.l1(xu)))
        x1 = F.relu(self.bn2(self.l2(x1)))
        x1 = self.l3(x1)

        x2 = F.relu(self.bn3(self.l4(xu)))
        x2 = F.relu(self.bn4(self.l5(x2)))
        x2 = self.l6(x2)
        return x1, x2



class TD3(object):
    """Agent class that handles the training of the networks and provides outputs as actions
    
        Args:
            state_dim (int): state size
            action_dim (int): action size
            max_action (float): highest action to take
            device (device): cuda or cpu to process tensors
            env (env): gym environment to use
    
    """
    
    def __init__(self, state_dim, action_dim, max_action, env):
        
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.ep_num = 0

        self.lr = 1e-3

        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr, weight_decay = 0.1)

        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr, weight_decay = 0.1)

        try:
            self.load()
        except Exception as e:
            print("failed to load checkpoint",e)

        for param in self.actor.parameters():
            param.requires_grad = True

        for param in self.critic.parameters():
            param.requires_grad = True

        self.max_action = max_action
        self.env = env


    def adjust_learning_rate(self, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        self.lr = max(self.lr * (0.3333333 ** (epoch // 10)),(10**-5))
        for param_group in self.critic_optimizer.param_groups:
            param_group['lr'] = self.lr

        for param_group in self.actor_optimizer.param_groups:
            param_group['lr'] = self.lr

        
    def select_action(self, state, noise=0.1):
        """Select an appropriate action from the agent policy
        
            Args:
                state (array): current state of environment
                noise (float): how much noise to add to acitons
                
            Returns:
                action (float): action clipped within action range
        
        """

        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        
        action = self.actor(state).cpu().data.cpu().numpy().flatten()

        if noise != 0: 
            action = (action + np.random.normal(0, noise, size=self.env.action_space[0]))
            
        return action.clip(self.env.action_low, self.env.action_high)

    def eval_mode(self):
        self.actor.eval()

    def train_mode(self):
        self.actor.train()

    
    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        """Train and update actor and critic networks
        
            Args:
                replay_buffer (ReplayBuffer): buffer for experience replay
                iterations (int): how many times to run training
                batch_size(int): batch size to sample from replay buffer
                discount (float): discount factor
                tau (float): soft update for main networks to target networks
                
            Return:
                actor_loss (float): loss from actor network
                critic_loss (float): loss from critic network
        
        """
        
        for it in range(iterations):

            # Sample replay buffer 
            x, y, u, r, d = replay_buffer.sample(batch_size)
            state = torch.FloatTensor(x).to(device)
            action = torch.FloatTensor(u).to(device)
            next_state = torch.FloatTensor(y).to(device)
            done = torch.FloatTensor(1 - d).to(device)
            reward = torch.FloatTensor(r).to(device)

            # Select action according to policy and add clipped noise 
            noise = torch.FloatTensor(u).data.normal_(0, policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (done * discount * target_Q).detach()

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(state, action)
            simple_q1 = current_Q1.detach().mean().cpu().numpy()
            # log_file.write("Q1 " + str(simple_q1) + "\n")

            simple_q2 = current_Q2.detach().mean().cpu().numpy()
            # log_file.write("Q2 " + str(simple_q2) + "\n")
            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q) 
            log_c_loss = critic_loss.detach().mean().cpu().numpy()
            # log_file.write("CL " + str(log_c_loss) + "\n")

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Delayed policy updates
            if it % policy_freq == 0:

                # Compute actor loss
                # actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
                comb_action = self.actor(state)
                # comb_action = torch.cat([vec, vel],1)
                q1, _ = self.critic(state,comb_action)
                actor_loss = q1.mean()

                log_a_loss = actor_loss.detach().cpu().numpy()


                # log_file.write("AL "+str(log_a_loss)+"\n")

                # Optimize the actor 
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Update the frozen target models
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


    def save(self, episode_number, filename, directory):
        torch.save({
            "ep": episode_number,
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_optim": self.actor_optimizer.state_dict(),
            "critic_optim" : self.critic_optimizer.state_dict(),
            },"./saves/model.pth")


    def load(self, filename="best_avg", directory="./saves"):
        print("Loading model ....")

        checkpoint = torch.load("./saves/model.pth")
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optim"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optim"])
        self.ep_num = checkpoint["ep"]

        print("Model load complete")