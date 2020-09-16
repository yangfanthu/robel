import robel
import gym
import torch
import torch.nn as nn
import gym
import numpy as np
import os
import sys
import datetime
import argparse


from modules import *
import utils

if __name__ == "__main__":
	env = gym.make('DClawTurnFixed-v0')
	parser = argparse.ArgumentParser()
	parser.add_argument("--start-timesteps", type=int, default=1e4)
	parser.add_argument("--max-timesteps", type=int, default=3e6)
	parser.add_argument("--eval-freq", type = int, default = 2000)
	args = parser.parse_args()
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	state_dim = env.reset().shape[0]
	action_dim = env.action_space.sample().shape[0]
	max_action = env.action_space.high[0]

	ddpg = DDPG(state_dim = state_dim,
	action_dim = action_dim,
	replay_buffer = None,
	writer = None,
	max_action = max_action,
	device = device)
	ddpg.restore_model(448000)
	current_state = env.reset()
	env.render()

	while True:
		action = ddpg.select_action(current_state, 'test')
		next_state, reward, done, info = env.step(action)
		suc = info['score/success']
		current_state = next_state
		if done:
			current_state = env.reset()