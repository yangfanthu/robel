import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import utils
import random
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
		hidden = torch.cat((state,action), dim=-1)
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
		hidden = self.fc_4(hidden)
		action = torch.tanh(hidden) * self.max_action
		return action

class DDPG(object):
	def __init__(self, 
		state_dim:int,
		action_dim:int, 
		device, 
		writer,
		buffer_max_size=int(1e6), 
		gamma:float = 0.99, 
		batch_size:int = 64, 
		max_action:float = 1., 
		hidden_size: int = 256,
		tau = 0.005,
		variance:float = 0.1,
		save_freq:int = 8000,
		record_freq:int=100):

		self.batch_size = batch_size
		self.gamma = gamma
		self.max_action = max_action
		self.hidden_size = hidden_size
		self.replay_buffer = utils.ReplayBuffer(state_dim = state_dim, action_dim = action_dim, max_size=buffer_max_size, device=device)
		self.tau = tau
		self.device = device
		self.variance = variance
		self.action_dim = action_dim
		self.state_dim	= state_dim
		self.save_freq = save_freq
		self.record_freq = record_freq
		self.critic = Critic(state_dim = state_dim, action_dim = action_dim, hidden_size = hidden_size).to(device)
		self.critic_target = copy.deepcopy(self.critic).to(device)
		self.critic_target.eval()
		self.actor = Actor(state_dim = state_dim, action_dim = action_dim, hidden_size = hidden_size, max_action = max_action).to(device)
		self.actor_target = copy.deepcopy(self.actor).to(device)
		self.actor_target.eval()

		self.mse_loss = nn.MSELoss()
		self.index = 0
		self.writer = writer
		learning_rate = 1e-3
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr = 1e-3)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr = 1e-3)

	def add_buffer(self,current_state,action,next_state,reward,done):
		self.replay_buffer.add(current_state,action,next_state,reward,done)
	def select_action(self, state, mode = "train"):
		temp_state = state
		state = torch.tensor(state).float().to(self.device)
		with torch.no_grad():
			if mode == "train":
				action = self.actor(state).cpu().detach().numpy()
				noise = np.random.normal(0,self.variance,self.action_dim)
				action = action + noise
			elif mode == "test":
				action = self.actor(state).cpu().detach().numpy()
		action = np.clip(action, -self.max_action, self.max_action)
		return action
	def save_model(self):
		print('saving...')
		torch.save(self.actor.cpu().state_dict(),'./saved_models/actor_{:07d}.ckpt'.format(self.index))
		torch.save(self.critic.cpu().state_dict(),'./saved_models/critic_{:07d}.ckpt'.format(self.index))
		torch.save(self.actor_target.state_dict(),'./saved_models/actor_target_{:07d}.ckpt'.format(self.index))
		torch.save(self.critic_target.state_dict(),'./saved_models/critic_target_{:07d}.ckpt'.format(self.index))
		self.actor.to(self.device)
		self.critic.to(self.device)
		print('finish saving')
	def restore_model(self, index):
		self.index = index
		self.actor.load_state_dict(torch.load('./saved_models/actor_{:07d}.ckpt'.format(index)))
		self.critic.load_state_dict(torch.load('./saved_models/critic_{:07d}.ckpt'.format(index)))
		print('finish restoring model')

	def train(self):
		self.index += 1
		current_state, action, next_state, reward, not_done = self.replay_buffer.sample(self.batch_size)
		
		self.critic_optimizer.zero_grad()
		with torch.no_grad():
			target = reward + self.gamma * not_done * self.critic_target(next_state, self.actor_target(next_state))
		current_q = self.critic(current_state, action)
		critic_loss = self.mse_loss(target, current_q)
		

		critic_loss.backward()
		self.critic_optimizer.step()

		self.actor_optimizer.zero_grad()
		actor_loss = - self.critic(current_state, self.actor(current_state)).mean()
		actor_loss.backward()
		self.actor_optimizer.step()

		if self.index % self.record_freq == 0:
			self.writer.add_scalar('./train/critic_loss',critic_loss.item(), self.index)
			self.writer.add_scalar('./train/actor_loss', actor_loss.item(), self.index)
			self.writer.add_scalar('./train/current_q', current_q.mean().item(), self.index)
			self.writer.add_scalar('./train/reward_max', reward.max().item(), self.index)
			self.writer.add_scalar('./train/reward_mean', reward.mean().item(), self.index)
			self.writer.add_scalar('./train/actor_q', -actor_loss.item(), self.index)


		for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

		for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

		if self.index % self.save_freq == 0:
			self.save_model()

class AdversarialDQN(object):
	def __init__(self, 
		state_dim, n_actions, 
		device, 
		writer, 
		buffer_max_size=int(1e6),
		epsilon=0.1, 
		hidden_size=256, 
		batch_size=64, 
		learning_rate=1e-3,
		gamma=0.9,
		tau=5e-3,
		save_freq=2000,
		record_freq=100):

		self.state_dim = state_dim # state space is the same as the ddpg agent
		self.n_actions = n_actions
		self.hidden_size = hidden_size
		self.q_function = Critic(state_dim, 1, hidden_size=hidden_size).to(device)
		self.q_target = copy.deepcopy(self.q_function).to(device)
		self.epsilon = epsilon
		self.device = device
		self.writer = writer
		self.replay_buffer = utils.ReplayBuffer(state_dim = state_dim, action_dim = 1, max_size=buffer_max_size, device=device)
		self.index = 0
		self.batch_size = batch_size
		self.learning_rate = learning_rate
		self.optimizer = torch.optim.Adam(self.q_function.parameters(), lr = learning_rate)
		self.gamma = gamma
		self.tau = tau
		self.record_freq = record_freq
		self.save_freq = save_freq
		self.criterion = torch.nn.MSELoss()
	def add_buffer(self,current_state,action,next_state,reward,done):
		self.replay_buffer.add(current_state,action,next_state,reward,done)
	def select_action(self, state, mode = "train"):
		dice = random.random()
		if dice < self.epsilon:
			action = np.random.randint(low = 0,high = self.n_actions, size = 1)
		else:
			scores = [self.q_function(torch.tensor(action)).cpu().item() for action in range(n_actions)]
			scores = np.array(scores)
			action = np.where(scores == scores.max())[0]
		return action
	def __select_action_target(self, state):
		# scores = []
		# for current_action in range(self.n_actions):
		# 	batch_action = np.ones((self.batch_size, 1)) * current_action
		# 	batch_action = torch.tensor(batch_action).float()
		# 	current_score = self.q_function(state, batch_action).cpu()
		# 	scores.append(current_score)
		# pdb.set_trace()
		scores = [np.array(self.q_function(state, torch.tensor(np.ones((self.batch_size, 1)) * action).float()).cpu())for action in range(self.n_actions)]
		scores = np.array(scores)
		scores = scores.transpose(1,0,2)
		max_scores = scores.max(1)
		action_list = []
		for i in range(self.batch_size):
			action = np.where(scores[i]==max_scores[i])[0]
			action_list.append(action)
		action = np.array(action_list)
		action = torch.tensor(action).float().to(self.device)
		return action
	def save_model(self):
		print("saving the adversarial q function....")
		torch.save(self.q_function.cpu().state_dict(),'./saved_models/advesaral_q_{:07d}.ckpt'.format(self.index))
		torch.save(self.q_function_target.cpu().state_dict(),'./saved_models/advesaral_q_target_{:07d}.ckpt'.format(self.index))
		self.q_function.to(self.device)
		print("finish saving the advesarial model")
	def restore_model(self, index):
		self.index = index
		self.q_function.load_state_dict(torch.load('./saved_models/advesaral_q_{:07d}.ckpt'.format(self.index)))
		self.q_target.load_state_dict(torch.load('./saved_models/advesaral_q_target_{:07d}.ckpt'.format(self.index)))
		print("finish restoring the advesarial model")
	def train(self):
		self.index += 1
		current_state, action, next_state, reward, not_done = self.replay_buffer.sample(self.batch_size)

		self.optimizer.zero_grad()
		
		with torch.no_grad():
			target = reward + self.gamma * not_done * self.q_target(next_state, self.__select_action_target(next_state))
		current_q = self.q_function(current_state, action)
		loss = self.criterion(target, current_q)

		loss.backward()
		self.optimizer.step()
		
		if self.index % self.record_freq == 0 and self.writer != None:
			self.writer.add_scalar('./train/adversarial_loss', loss.cpu().item(), self.index)
			self.writer.add_scalar('./train/advesarial_q', current_q.cpu().mean(), self.index)

		for param, target_param in zip(self.q_function.parameters(), self.q_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

		if self.index % self.save_freq == 0:
			self.save_model()

if __name__ == "__main__":
	adversarial_dqn = AdversarialDQN(state_dim=3, n_actions=3,device='cpu',writer=None)
	dumb_state = np.ones(3)
	dumb_next_state = np.ones(3)
	dumb_action = np.array([1])
	dumb_reward = 10
	adversarial_dqn.add_buffer(dumb_state, dumb_action, dumb_next_state, dumb_reward, False)
	adversarial_dqn.train()
	print('finish')


		




