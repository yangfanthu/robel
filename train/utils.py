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


class RDPGReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6), max_episode_timesteps=40, device=torch.device('cuda')):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = torch.zeros((max_size, max_episode_timesteps, state_dim), device=device)
		self.action = torch.zeros((max_size, max_episode_timesteps, action_dim), device=device)
		self.next_state = torch.zeros((max_size, max_episode_timesteps, state_dim), device=device)
		self.reward = torch.zeros((max_size, max_episode_timesteps, 1), device=device)
		self.not_done = torch.zeros((max_size, max_episode_timesteps, 1), device=device)
		self.episode_timesteps = torch.zeros((max_size))

		self.device = device


	def add(self, state, action, next_state, reward, done, episode_timesteps):
		self.state[self.ptr] = torch.tensor(state, device=self.device)
		self.action[self.ptr] = torch.tensor(action, device=self.device)
		self.next_state[self.ptr] = torch.tensor(next_state, device=self.device)
		self.reward[self.ptr] = torch.tensor(reward, device=self.device)
		self.not_done[self.ptr] = torch.tensor(1. - done, device=self.device)
		self.episode_timesteps[self.ptr] = episode_timesteps

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)
		return (
			self.state[ind],
			self.action[ind],
			self.next_state[ind],
			self.reward[ind],
			self.not_done[ind],
			self.episode_timesteps[ind]
		)
def generate_mask(episode_timestep, max_episode_timesteps):
	#episode_timestep [batchsize, timestep]
	batch_size = episode_timestep.shape[0]
	mask = torch.ones((batch_size, max_episode_timesteps))
	for i in range(batch_size):
		if episode_timestep[i] < max_episode_timesteps:
			mask[i,int(episode_timestep[i].item()):] = 0
	return mask  # shape:[batch, max time steps]