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
import random


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
	buffer_max_size=int(1e6),
	writer = None,
	max_action = max_action,
	device = device,
	hidden_size=512)
	ddpg.restore_model(1505000)
	adversary = AdversarialDQN(state_dim, action_dim, device, writer=None,buffer_max_size=int(1e6))
	adversary.restore_model(1505000)
	current_state = env.reset()
	env.render()
	sum_reward = 0
	index = 0
	episode = 0
	# env._max_episode_steps = 60
	with torch.no_grad():
		while True:
			action = ddpg.select_action(current_state, 'test')
			adversary_action = adversary.select_action(current_state,'test')
			index += 1
			# action = np.ones(9) * -0.6
			# action[4] = -0.6  # this case doesn't work, also 6,7
			action[8] = -0.6

			# action[random.randint(0,8)] = 0
			# print(adversary_action)
			# action[adversary_action[0]] = -0.6
			# action[random.randint(0,8)] = 0
			next_state, reward, done, info = env.step(action)
			sum_reward += reward 
			# print(sum_reward)
			suc = info['score/success']
			
			current_state = next_state
			if done:
				episode += 1
				current_state = env.reset()
				print(sum_reward)
				# random_list.append(sum_reward)
				# print("episode: {}, avg: {}".format(episode, np.array(random_list).mean()))
				sum_reward = 0
			