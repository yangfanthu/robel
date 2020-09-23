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

class AdversarialEnv(object):
	def __init__(self,
		ddpg_action_dim,
		ddpg_state_dim,
		ddpg_buffer_max_size,
		writer,
		ddpg_gamma,
		ddpg_hidden_size,
		ddpg_save_freq,
		ddpg_record_freq,
		ddpg_batch_size,
		ddpg_max_action,
		ddpg_tau,
		ddpg_variance,
		device,
		env_name='DClawTurnFixed-v0',
		real_robot=False):
		self.ddpg_action_dim = ddpg_action_dim
		self.ddpg_state_dim = ddpg_state_dim
		self.ddpg_buffer_max_size = ddpg_buffer_max_size
		self.writer = writer
		self.ddpg_gamma = ddpg_gamma
		self.ddpg_hidden_size = ddpg_hidden_size
		self.ddpg_save_freq = ddpg_save_freq
		self.device = device
		self.env_name = env_name 
		self.real_robot = real_robot
		self.ddpg = DDPG(state_dim=ddpg_state_dim,
						action_dim=ddpg_action_dim,
						device=device,
						writer=writer,
						buffer_max_size=ddpg_buffer_max_size,
						gamma=ddpg_gamma,
						save_freq=ddpg_save_freq,
						record_freq=ddpg_record_freq)
		if real_robot:
			self.base_env = gym.make(env_name, device_path='/dev/tty.usbserial-FT3WI485')
		else:
			self.base_env = gym.make(env_name)
		self.action_space = self.base_env.action_space
	def step(self, advesarial_action:int, ddpg_obs):
		ddpg_action = self.ddpg.select_action(ddpg_obs, 'test')
		ddpg_action[advesarial_action] = 0.
		next_state, reward, done, info  = self.base_env.step(ddpg_action)
		return next_state, reward, done, info
	def seed(self, input_seed):
		self.base_env.seed(input_seed)
	def reset(self):
		return self.base_env.reset()



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--start-timesteps", type=int, default=int(1e4))
	parser.add_argument("--max-timesteps", type=int, default=int(3e6))
	parser.add_argument("--eval-freq", type=int, default=5000)
	parser.add_argument("--save-freq", type=int, default=5000)
	parser.add_argument("--seed", type=int, default=0)
	parser.add_argument("--buffer-max-size",type=int,default=int(1e6))
	parser.add_argument("--ddpg-training-steps", type=int, default=int(1e3))
	parser.add_argument("--adversary-training-steps", type=int,default=int(1e3))
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
	adversary = AdversarialDQN(state_dim, action_dim, device, writer,args.buffer_max_size)
    
	
	"advesarial agent code"
	adversarial_env = AdversarialEnv(ddpg_action_dim = action_dim,
									ddpg_state_dim=state_dim,
									ddpg_buffer_max_size=args.buffer_max_size,
									writer=writer,
									device=device,
									ddpg_gamma=0.9,
									ddpg_hidden_size=256,
									ddpg_save_freq=5000,
									ddpg_record_freq=5000,
									ddpg_batch_size=64,
									ddpg_max_action=max_action,
									ddpg_tau=5e-3,
									ddpg_variance=0.1)
	current_state = adversarial_env.reset()
	episode = 0
	t = 0
	ddpg_t = 0
	adversary_t = 0
	while True:
		if t > args.max_timesteps:
			break
		for i in range(args.ddpg_training_steps):
			t += 1
			ddpg_t += 1
			
			if t % args.eval_freq == 0:
				print("-------------------------------------------")
				print("steps:{:07d}".format(t))
				print("episode:{:07d}".format(episode))
				avg_reward = eval_policy(adversarial_env.ddpg, 'DClawTurnFixed-v0')
				writer.add_scalar('/eval/avg_reward',avg_reward, t)
			
			
			if ddpg_t == args.start_timesteps:
				print("start ddpg learning")
			if ddpg_t < args.start_timesteps:
				action = adversarial_env.action_space.sample()
			else:
				action = adversarial_env.ddpg.select_action(current_state, 'train')
			adversary_action = adversary.select_action(current_state,'test')
			action[adversary_action[0]] = 0
			next_state, reward, done, info = adversarial_env.base_env.step(action)
			suc = info['score/success']
			adversarial_env.ddpg.add_buffer(current_state,action,next_state,reward,done)
			if ddpg_t > args.start_timesteps:
				adversarial_env.ddpg.train()
			current_state = next_state
			if done:
				current_state = adversarial_env.reset()
				episode += 1
		current_state = adversarial_env.reset()
		for i in range(args.adversary_training_steps):
			t += 1
			adversary_t += 1
			action = adversary.select_action(current_state,'train')
			next_state, reward, done, info = adversarial_env.step(action[0],current_state)
			adversary.add_buffer(current_state, action, next_state, reward, done)
			if adversary_t == args.start_timesteps:
				print("start training the adversary!")
			if adversary_t > args.start_timesteps:
				adversary.train()
			current_state = next_state

			if t % args.eval_freq == 0:
				print("-------------------------------------------")
				print("steps:{:07d}".format(t))
				print("episode:{:07d}".format(episode))
				avg_reward = eval_policy(adversarial_env.ddpg, 'DClawTurnFixed-v0')
				writer.add_scalar('/eval/avg_reward',avg_reward, t)
				
			if done:
				current_state = adversarial_env.reset()
				episode += 1



			
"precious pure ddpg code"
	# ddpg = DDPG(state_dim=state_dim,
	# action_dim=action_dim,
	# buffer_max_size=args.buffer_max_size,
	# writer=writer,
	# gamma=0.9,
	# max_action=max_action,
	# device=device,
	# hidden_size=512,
	# save_freq=args.save_freq)
	# current_state = base_env.reset()
	# episode = 0
	# for t in range(args.max_timesteps):
	# 	if t == args.start_timesteps:
	# 		print("start learning")
	# 	if t % args.eval_freq== 0:
	# 		print("-------------------------------------------")
	# 		print("steps:{:07d}".format(t))
	# 		print("episode:{:07d}".format(episode))
	# 		avg_reward = eval_policy(ddpg, 'DClawTurnFixed-v0')
	# 		writer.add_scalar('/eval/avg_reward',avg_reward, t)
	# 	if t < args.start_timesteps:
	# 		action = base_env.action_space.sample()
	# 	else:
	# 		action = ddpg.select_action(current_state, 'train')
	# 	next_state, reward, done, info = base_env.step(action)
	# 	suc = info['score/success']
	# 	ddpg.add_buffer(current_state,action,next_state,reward,done)
	# 	if t > args.start_timesteps:
	# 		ddpg.train()
	# 	current_state = next_state
	# 	if done:
	# 		current_state = base_env.reset()
	# 		episode += 1