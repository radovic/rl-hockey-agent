import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils import ReplayBuffer, OrnsteinUhlenbeckNoise
from networks import BasicActor, BasicCritic, MirrorActor, MirrorCritic

class DDPG():
    def __init__(self,
                 action_space_dim,
                 state_space_dim,
                 max_action,
                 use_mirror=True,
                 lr_actor=3e-4,
                 lr_critic=3e-4,
                 discount_rate=0.95,
                 batch_size=128,
                 max_buffer_size=50000,
                 soft_update_ts=1,
                 tau=0.005,
                 dr3_coeff=0,
                 use_resets=False,
                 save_path='./',
                 logger=None
                ):
        self.identifier = 'DDPG'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.discount_rate = discount_rate
        self.batch_size = batch_size
        self.tau = tau
        self.dr3_coeff = dr3_coeff
        self.max_action = max_action
        self.state_space_dim = state_space_dim
        self.action_space_dim = action_space_dim
        self.lr_critic = lr_critic
        self.lr_actor = lr_actor
        self.use_mirror = use_mirror
        self.logger = logger

        self.initialize_networks()
        self.use_resets = use_resets
        self.save_path = save_path

        self.noise = OrnsteinUhlenbeckNoise(mu=np.zeros(action_space_dim))
        self.replay_buffer = ReplayBuffer(state_space_dim, action_space_dim, max_buffer_size)
        self.soft_update_ts = soft_update_ts
        self.last_soft_update = 0
        self.reset_number = 0
        self.reset_interval = 2e5
        self.time_step = 0

    def initialize_networks(self):
        Critic, Actor = BasicCritic, BasicActor
        if self.use_mirror:
            Critic = MirrorCritic
            Actor = MirrorActor
        self.critic = Critic(self.state_space_dim, self.action_space_dim).to(self.device)
        self.critic_target = Critic(self.state_space_dim, self.action_space_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer  = optim.NAdam(self.critic.parameters(), lr=self.lr_critic)
        self.actor = Actor(self.state_space_dim, self.action_space_dim, self.max_action).to(self.device)
        self.actor_target = Actor(self.state_space_dim, self.action_space_dim, self.max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.NAdam(self.actor.parameters(), lr=self.lr_actor)
        
        
    def reset(self, save=True):
        if self.logger != None:
            self.logger.register('Resetting networks...')
        self.reset_number += 1
        if save:
            self.save_model(f'reset-{self.reset_number}')
        self.last_soft_update = 0
        self.time_step = 0
        self.initialize_networks()
        
    
    def optimize_critic(self, s, a, r, s_prime, done_mask):
        a_prime = self.actor_target(s_prime)
        # Q_target, ll_features_prime = self.critic_target(s_prime, a_prime)
        Q_target, _ = self.critic_target(s_prime, a_prime)
        _, ll_features_prime = self.critic(s_prime, a_prime.detach())
        td_target = r + self.discount_rate * Q_target * done_mask
        Q_current, ll_features_current = self.critic(s, a)
        dr3_regularizer = 0
        if self.dr3_coeff != 0:
            dr3_regularizer = self.dr3_coeff *\
                (ll_features_current * ll_features_prime).sum(-1).mean()
        critic_loss = F.mse_loss(Q_current, td_target.detach()) + dr3_regularizer
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        return critic_loss.detach()
        
    def optimize_actor(self, s):
        actor_loss = -self.critic(s,self.actor(s))[0].mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        return actor_loss.detach()
    
    def soft_update(self, net, net_target):
        for param_target, param in zip(net_target.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)
    
    def save_model(self, name, save_model_path=None):
        if save_model_path == None: save_model_path = self.save_path
        actor_path = os.path.join(save_model_path, 'actor')
        critic_path = os.path.join(save_model_path, 'critic')
        if not os.path.exists(actor_path): os.makedirs(actor_path)
        if not os.path.exists(critic_path): os.makedirs(critic_path)
        actor_path = os.path.join(actor_path, name + '.pt')
        critic_path = os.path.join(critic_path, name + '.pt')
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)
        
    def load_models(self, actor_model_path, critic_model_path):
        self.actor.load_state_dict(torch.load(actor_model_path, map_location=self.device))
        self.critic.load_state_dict(torch.load(critic_model_path, map_location=self.device))
    
    def train(self, fit_iter=32):
        actor_losses, critic_losses = [], []
        for t in range(fit_iter):
            s,a,r,s_prime,done_mask  = self.replay_buffer.sample(self.batch_size)
            critic_losses.append(self.optimize_critic(s, a, r, s_prime, done_mask))
            actor_losses.append(self.optimize_actor(s))
            self.time_step += 1
            if self.use_resets and self.time_step >= self.reset_interval:
                self.reset()
            if self.time_step - self.last_soft_update == self.soft_update_ts:
                self.soft_update(self.actor, self.actor_target)
                self.soft_update(self.critic,  self.critic_target)
                self.last_soft_update = self.time_step
        return 1, 1
    
    def act(self, state, eps=0):
        a = self.actor(torch.FloatTensor(state).to(self.device))
        a = a.detach().cpu().numpy() + eps * self.noise()
        a = np.clip(a, -1 * self.max_action, self.max_action)
        return a

    def remote_act(self, state):
        return self.act(state)

    def before_game_starts(self) -> None:
        pass

    def after_game_ends(self) -> None:
        pass
