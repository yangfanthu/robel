import argparse
import logging
import sys

import robel
import gym
import gym.wrappers
import numpy as np
import torch
from torch import nn

import pfrl
from pfrl.agents.ddpg import DDPG
from pfrl import experiments
from pfrl import explorers
from pfrl import utils
from pfrl import replay_buffers
from pfrl.nn import ConcatObsAndAction
from pfrl.nn import BoundByTanh
from pfrl.policies import DeterministicHead

import pdb


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument(
	    "--outdir",
	    type=str,
	    default="results",
	    help=(
	        "Directory path to save output files."
	        " If it does not exist, it will be created."
	    ),
	)
	parser.add_argument(
	    "--env",
	    type=str,
	    default="'DClawTurnFixed-v0'",
	    help="OpenAI Gym MuJoCo env to perform algorithm on.",
	)
	parser.add_argument("--seed", type=int, default=0, help="Random seed [0, 2 ** 32)")
	parser.add_argument(
	    "--gpu", type=int, default=-1, help="GPU to use, set to -1 if no GPU."
	)
	parser.add_argument(
	    "--load", type=str, default="", help="Directory to load agent from."
	)
	parser.add_argument(
	    "--steps",
	    type=int,
	    default=10 ** 6,
	    help="Total number of timesteps to train the agent.",
	)
	parser.add_argument(
	    "--eval-n-runs",
	    type=int,
	    default=10,
	    help="Number of episodes run for each evaluation.",
	)
	parser.add_argument(
	    "--eval-interval",
	    type=int,
	    default=5000,
	    help="Interval in timesteps between evaluations.",
	)
	parser.add_argument(
	    "--replay-start-size",
	    type=int,
	    default=10000,
	    help="Minimum replay buffer size before " + "performing gradient updates.",
	)
	parser.add_argument("--batch-size", type=int, default=64, help="Minibatch size")
	parser.add_argument(
	    "--render", action="store_true", help="Render env states in a GUI window."
	)
	parser.add_argument(
	    "--demo", action="store_true", help="Just run evaluation, not training."
	)
	parser.add_argument("--load-pretrained", action="store_true", default=False)
	parser.add_argument(
	    "--pretrained-type", type=str, default="best", choices=["best", "final"]
	)
	parser.add_argument(
	    "--monitor", action="store_true", help="Wrap env with gym.wrappers.Monitor."
	)
	parser.add_argument(
	    "--log-level", type=int, default=logging.INFO, help="Level of the root logger."
	)
	args = parser.parse_args()

	logging.basicConfig(level=args.log_level)

	args.outdir = './results'
	print("Output files are saved in {}".format(args.outdir))

	# Set a random seed used in PFRL
	utils.set_random_seed(args.seed)

	def make_env(test):
	    env = gym.make('DClawTurnFixed-v0')
	    # Unwrap TimeLimit wrapper
	    assert isinstance(env, gym.wrappers.TimeLimit)
	    env = env.env
	    # Use different random seeds for train and test envs
	    env_seed = 2 ** 32 - 1 - args.seed if test else args.seed
	    env.seed(env_seed)
	    # Cast observations to float32 because our model uses float32
	    env = pfrl.wrappers.CastObservationToFloat32(env)
	    if args.monitor:
	        env = pfrl.wrappers.Monitor(env, args.outdir)
	    if args.render and not test:
	        env = pfrl.wrappers.Render(env)
	    return env

	# env = make_env(test=False)
	env = gym.make('DClawTurnFixed-v0')
	timestep_limit = env.spec.max_episode_steps
	obs_space = env.observation_space
	action_space = env.action_space
	print("Observation space:", obs_space)
	print("Action space:", action_space)

	obs_size = obs_space.low.size
	action_size = action_space.low.size

	q_func = nn.Sequential(
        ConcatObsAndAction(),
        nn.Linear(obs_size + action_size, 400),
        nn.ReLU(),
        nn.Linear(400, 300),
        nn.ReLU(),
        nn.Linear(300,300),
        nn.ReLU(),
        nn.Linear(300, 1),
    )
    policy = nn.Sequential(
        nn.Linear(obs_size, 400),
        nn.ReLU(),
        nn.Linear(400, 300),
        nn.ReLU(),
        nn.Linear(300,300),
        nn.ReLU(),
        nn.Linear(300, action_size),
        BoundByTanh(low=action_space.low, high=action_space.high),
        DeterministicHead(),
    )
	model = nn.ModuleList([policy, q_func])
	model.load_state_dict("./results/best/model.pt")
	policy = model[0]

	print("finish loading")


	for i_episode in range(20):
		observation = env.reset()
		for t in range(100):
			env.render()
			observation = torch.tensor(observation).float()
			action = policy(observation)
			action = np.array(action)
			observation, reward, done, info = env.step(action)
			if done:
				print("Episode finished after {} timesteps".format(t+1))
				break
	env.close()
