"the code only with ddpg, in order to check the difference between my code and pfrl"
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

from tensorboardX import SummaryWriter


import pdb

def eval_policy(policy, env_name, eval_episodes=10, real_robot = False, seed = 0):
	env_seed = 2 ** 32 - 1 - seed
	if real_robot:
		eval_env = gym.make(env_name, device_path='/dev/tty.usbserial-FT3WI485')
	else:
		eval_env = gym.make(env_name)
	eval_env.seed(env_seed)

	avg_reward = 0.
	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		while not done:
			action = policy.select_action(np.array(state), 'test')
			state, reward, done, _ = eval_env.step(action)
			avg_reward += reward
		
	avg_reward /= eval_episodes

	print("---------------------------------------")
	print("Evaluation over {} episodes: {:.3f}".format(eval_episodes, avg_reward))
	print("---------------------------------------")
	return avg_reward

base_env = gym.make('DClawTurnFixed-v0')
# env = gym.make('DClawTurnFixed-v0', device_path='/dev/tty.usbserial-FT3WI485')


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--start-timesteps", type=int, default=1e4)
	parser.add_argument("--max-timesteps", type=int, default=3e6)
	parser.add_argument("--eval-freq", type = int, default = 5000)
	parser.add_argument("--save-freq", type = int, default = 5000)
	parser.add_argument("--seed", type = int, default=0)
	args = parser.parse_args()
	base_env.seed(args.seed)
	if not os.path.exists('./logs'):
		os.system('mkdir logs')
	if not os.path.exists('./saved_models'):
		os.system('mkdir saved_models')
	writer = SummaryWriter(logdir=('logs/{}').format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	state_dim = base_env.reset().shape[0]
	action_dim = base_env.action_space.sample().shape[0]
	max_action = base_env.action_space.high[0]

	replay_buffer = utils.ReplayBuffer(state_dim = state_dim, action_dim = action_dim, max_size=int(1e6), device=device)
	ddpg = DDPG(state_dim = state_dim,
	action_dim = action_dim,
	replay_buffer = replay_buffer,
	writer = writer,
	gamma = 0.9,
	max_action = max_action,
	device = device,
	hidden_size=512,
	save_freq=args.save_freq)
	current_state = base_env.reset()
	episode = 0
	for t in range(int(args.max_timesteps)):
		if t == int(args.start_timesteps):
			print("start learning")
		if t % int(args.eval_freq) == 0:
			print("-------------------------------------------")
			print("steps:{:07d}".format(t))
			print("episode:{:07d}".format(episode))
			avg_reward = eval_policy(ddpg, 'DClawTurnFixed-v0')
			writer.add_scalar('/eval/avg_reward',avg_reward, t)
		if t < int(args.start_timesteps):
			action = base_env.action_space.sample()
		else:
			action = ddpg.select_action(current_state, 'train')
		next_state, reward, done, info = base_env.step(action)
		suc = info['score/success']
		replay_buffer.add(current_state,action,next_state,reward,done)
		if t > int(args.start_timesteps):
			ddpg.train()
		current_state = next_state
		if done:
			current_state = base_env.reset()
			episode += 1
