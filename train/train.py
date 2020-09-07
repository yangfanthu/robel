import robel
import gym
import pfrl
import torch
import torch.nn as nn
import gym
import numpy as np

import logging
import sys

from modules import *

import pdb

env = gym.make('DClawTurnFixed-v0')
# env = gym.make('DClawTurnFixed-v0', device_path='/dev/tty.usbserial-FT3WI485')
def burnin_action_func():
	"""Select random actions until model is updated one or more times."""
	return np.random.uniform(env.action_space.low, env.action_space.high).astype(np.float32)


if __name__ == "__main__":
	logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='')
	# Create a simulation environment for the D'Claw turn task.
	# env = gym.make('DClawTurnFixed-v0')

	# Create a hardware environment for the D'Claw turn task.
	# `device_path` refers to the device port of the Dynamixel USB device.
	# e.g. '/dev/ttyUSB0' for Linux, '/dev/tty.usbserial-*' for Mac OS.
	# env = gym.make('DClawTurnFixed-v0', device_path='/dev/ttyUSB0')

	# Reset the environent and perform a random action.
	obs = env.reset()
	observation_dim = obs.shape[0]
	action_dim = env.action_space.sample().shape[0]

	q_function = QFunction(observation_dim = observation_dim, action_dim = action_dim)
	policy = Policy(observation_dim = observation_dim, action_dim = action_dim, max_action = 0.1)

	actor_optimizer = torch.optim.Adam(q_function.parameters(), eps=1e-2)
	critic_optimizer = torch.optim.Adam(policy.parameters(), eps=1e-2)
	replay_buffer = pfrl.replay_buffers.ReplayBuffer(capacity=10 ** 6)
	explorer = pfrl.explorers.AdditiveGaussian(scale=0.1, low=env.action_space.low, high=env.action_space.high)
	# explorer = pfrl.explorers.ConstantEpsilonGreedy(epsilon=0.3, random_action_func=env.action_space.sample)
	gamma = 0.9
	batch_size = 64
	gpu = -1
	replay_start_size = 500
	steps = 10**6
	phi = lambda x: x.astype(np.float32, copy=False)

	agent = pfrl.agents.DDPG(policy,
		q_function,
		actor_optimizer,
		critic_optimizer,
		replay_buffer,
		gamma=0.99,
		explorer=explorer,
		replay_start_size=replay_start_size,
		target_update_method="soft",
		target_update_interval=1,
		update_interval=1,
		soft_update_tau=5e-3,
		n_times_update=1,
		gpu = gpu,
		minibatch_size=batch_size,
		burnin_action_func=burnin_action_func,
		phi = phi)
	env.render()
	pfrl.experiments.train_agent_with_evaluation(
            agent=agent,
            env=env,
            steps=steps,
            eval_env=env,
            eval_n_steps=None,
            eval_n_episodes=10,
            eval_interval=5000,
            outdir='./results',
            train_max_episode_len=env.spec.max_episode_steps,
        )
	print('Finished.')
	# pfrl.experiments.train_agent_with_evaluation(
	#     agent,
	#     env,
	#     steps=2000,           # Train the agent for 2000 steps
	#     eval_n_steps=None,       # We evaluate for episodes, not time
	#     eval_n_episodes=10,       # 10 episodes are sampled for each evaluation
	#     train_max_episode_len=200,  # Maximum length of each episode
	#     eval_interval=1000,   # Evaluate the agent after every 1000 steps
	#     outdir='result',      # Save everything to 'result' directory
	# )
	# env.step(env.action_space.sample())

	# for _ in range(1000):
	#     env.render()
	#     env.step(env.action_space.sample()) # take a random action
	# env.close()