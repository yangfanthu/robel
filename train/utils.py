import torch
import numpy as np

class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6), device=torch.device('cuda')):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = torch.zeros((max_size, state_dim), device=device)
		self.action = torch.zeros((max_size, action_dim), device=device)
		self.next_state = torch.zeros((max_size, state_dim), device=device)
		self.reward = torch.zeros((max_size, 1), device=device)
		self.not_done = torch.zeros((max_size, 1), device=device)

		self.device = device


	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = torch.tensor(state, device=self.device)
		self.action[self.ptr] = torch.tensor(action, device=self.device)
		self.next_state[self.ptr] = torch.tensor(next_state, device=self.device)
		self.reward[self.ptr] = torch.tensor(reward, device=self.device)
		self.not_done[self.ptr] = torch.tensor(1. - done, device=self.device)

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)
		return (
			self.state[ind],
			self.action[ind],
			self.next_state[ind],
			self.reward[ind],
			self.not_done[ind]
		)