import torch
import torch.nn as nn
import torch.nn.functional as F
from pfrl.policies import DeterministicHead
from pfrl.nn import ConcatObsAndAction
import pdb

class QFunction(torch.nn.Module):
	def __init__(self, observation_dim:int , action_dim:int, hidden_size: int = 256):
		super(QFunction, self).__init__()
		self.observation_dim = observation_dim
		self.action_dim	= action_dim
		self.hidden_size = hidden_size
		self.concat_obs_and_action = ConcatObsAndAction()
		self.fc_1 = nn.Linear(observation_dim + action_dim, hidden_size)
		self.fc_2 = nn.Linear(hidden_size, hidden_size)
		self.fc_3 = nn.Linear(hidden_size, hidden_size)
		self.fc_4 = nn.Linear(hidden_size, 1)
	def forward(self, x):
		hidden = self.concat_obs_and_action(x)
		hidden = F.relu(self.fc_1(hidden))
		hidden = F.relu(self.fc_2(hidden))
		hidden = F.relu(self.fc_3(hidden))
		q_value = F.relu(self.fc_4(hidden))
		return q_value

class Policy(torch.nn.Module):
	def __init__(self, max_action:float, observation_dim: int, action_dim:int, hidden_size: int = 256):
		super(Policy, self).__init__()
		self.max_action = max_action
		self.observation_dim = observation_dim
		self.action_dim	= action_dim
		self.hidden_size = hidden_size

		self.fc_1 = nn.Linear(observation_dim, hidden_size)
		self.fc_2 = nn.Linear(hidden_size, hidden_size)
		self.fc_3 = nn.Linear(hidden_size, hidden_size)
		self.fc_4 = nn.Linear(hidden_size, action_dim)
		self.deterministic_head = DeterministicHead()
	def forward(self, x):
		hidden = F.relu(self.fc_1(x))
		hidden = F.relu(self.fc_2(hidden))
		hidden = F.relu(self.fc_3(hidden))
		hidden = F.relu(self.fc_4(hidden))
		action = torch.tanh(hidden) * self.max_action
		action = self.deterministic_head(action)
		return action
