import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import utils
import pdb

class Critic(torch.nn.Module):
	def __init__(self, state_dim:int , action_dim:int, hidden_size: int = 256):
		super(Critic, self).__init__()
		self.state_dim = state_dim
		self.action_dim	= action_dim
		self.hidden_size = hidden_size
		self.fc_1 = nn.Linear(state_dim + action_dim, hidden_size)
		self.fc_2 = nn.Linear(hidden_size, hidden_size)
		self.fc_3 = nn.Linear(hidden_size, hidden_size)
		self.fc_4 = nn.Linear(hidden_size, 1)
	def forward(self, state, action):
		hidden = torch.cat((state,action), dim = -1)
		hidden = F.relu(self.fc_1(hidden))
		hidden = F.relu(self.fc_2(hidden))
		hidden = F.relu(self.fc_3(hidden))
		q_value = self.fc_4(hidden)
		return q_value

class Actor(torch.nn.Module):
	def __init__(self, state_dim: int, action_dim:int, max_action:float = 1.,hidden_size: int = 256):
		super(Actor, self).__init__()
		self.max_action = max_action
		self.state_dim = state_dim
		self.action_dim	= action_dim
		self.hidden_size = hidden_size

		self.fc_1 = nn.Linear(state_dim, hidden_size)
		self.fc_2 = nn.Linear(hidden_size, hidden_size)
		self.fc_3 = nn.Linear(hidden_size, hidden_size)
		self.fc_4 = nn.Linear(hidden_size, action_dim)
	def forward(self, state):
		hidden = F.relu(self.fc_1(state))
		hidden = F.relu(self.fc_2(hidden))
		hidden = F.relu(self.fc_3(hidden))
		hidden = F.relu(self.fc_4(hidden))
		action = torch.tanh(hidden) * self.max_action
		return action

class DDPG(object):
	def __init__(self, 
		state_dim:int,
		action_dim:int, 
		replay_buffer: utils.ReplayBuffer,
		device, 
		writer, gamma:float = 0.9, 
		batch_size:int = 64, 
		max_action:float = 1., 
		hidden_size: int = 256,
		tau = 0.005,
		variance:float = 0.3,
		save_freq:int = 4000):

		self.batch_size = batch_size
		self.gamma = gamma
		self.max_action = max_action
		self.hidden_size = hidden_size
		self.replay_buffer = replay_buffer
		self.tau = tau
		self.device = device
		self.variance = variance
		self.action_dim = action_dim
		self.state_dim	= state_dim
		self.save_freq = save_freq
		self.critic = Critic(state_dim = state_dim, action_dim = action_dim, hidden_size = hidden_size)
		self.critic_target = copy.deepcopy(self.critic)
		self.actor = Actor(state_dim = state_dim, action_dim = action_dim, hidden_size = hidden_size, max_action = max_action)
		self.actor_target = copy.deepcopy(self.actor)
		self.mse_loss = nn.MSELoss()
		self.index = 0
		self.writer = writer
		learning_rate = 1e-3
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr = 1e-3)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr = 1e-3)

	def select_action(self, state, mode = "train"):
		state = torch.tensor(state).float().to(self.device)
		with torch.no_grad():
			if mode == "train":
				action = self.actor(state).detach().numpy()
				noise = np.random.normal(0,self.variance,self.action_dim)
				action = action + noise
			elif mode == "test":
				action = self.actor(state).detach().numpy()
		return action
	def save_model(self):
		print('saving...')
		torch.save(self.actor.cpu().state_dict(),'./saved_models/actor_{:07d}.ckpt'.format(self.index))
		torch.save(self.critic.cpu().state_dict(),'./saved_models/critic_{:07d}.ckpt'.format(self.index))
		print('finish saving')
	def restore_model(self, index):
		self.index = index
		self.actor.load_state_dict(torch.load('./saved_models/actor_{:07d}.ckpt'.format(index)))
		self.critic.load_state_dict(torch.load('./saved_models/critic_{:07d}.ckpt'.format(index)))
		print('finish restoring model')

	def train(self):
		self.index += 1
		current_state, action, next_state, reward, not_done = self.replay_buffer.sample(self.batch_size)
		with torch.no_grad():
			target = reward + self.gamma * not_done * self.critic_target(next_state, self.actor_target(next_state))
		current_q = self.critic(current_state, action)
		critic_loss = self.mse_loss(current_q, target)
		

		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		actor_loss = -self.critic(current_state, self.actor(current_state)).mean()
		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()

		self.writer.add_scalar('./train/critic_loss',critic_loss.cpu().item(), self.index)
		self.writer.add_scalar('./train/actor_loss', actor_loss.cpu().item(), self.index)
		self.writer.add_scalar('./train/current_q', current_q.cpu().mean().item(), self.index)
		self.writer.add_scalar('./train/reward_max', reward.max().cpu().item(), self.index)
		self.writer.add_scalar('./train/reward_mean', reward.mean().cpu().item(), self.index)
		self.writer.add_scalar('./train/actor_q', -actor_loss.cpu().item(), self.index)


		for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

		for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

		if self.index % self.save_freq == 0:
			self.save_model()







