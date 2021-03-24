import argparse
import os
import datetime
from typing import final
import gym
import numpy as np
import itertools
import torch
import robel
from sac import SAC
from torch.utils.tensorboard import SummaryWriter
from replay_memory import ReplayMemory
import imageio

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="DClawTurnFixed-v0",
                    help='Mujoco Gym environment (default: DClawTurnFixed-v0)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
parser.add_argument('--offline', action="store_true")
args = parser.parse_args()
# def save_gif(filename, frames, bounce=False, color_last=False, duration=0.05):
#     print('Save {} frames into video...'.format(len(frames)))
    # inputs = [np.array(frame.copy())/255 for frame in frames]
    # images = []
    # for tensor in inputs:
    #     if not color_last:
    #         tensor = tensor.transpose(0,1).transpose(1,2)
    #     tensor = np.clip(tensor, 0, 1)
    #     images.append((tensor * 255).astype('uint8'))
    # if bounce:
    #     images = images + list(reversed(images[1:-1]))
    # imageio.mimsave(filename, frames)

env = gym.make(args.env_name)
env.seed(args.seed)
env.action_space.seed(args.seed)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

if not os.path.exists('videos'):
    os.system('mkdir videos')
# Agent
agent = SAC(env.observation_space.shape[0], env.action_space, args)
agent.load_model("/home/evan/github/robel/class_project/models/2021-03-21_19-22-32_SAC_DClawTurnFixedD0-v0/sac_actor_1200")

#Tesnorboard
# Memory

# Training Loop
total_numsteps = 0
updates = 0

for i_episode in range(10):
    state = env.reset()
    episode_reward = 0
    done = False
    while not done:
    
        env.render()
        action = agent.select_action(state, evaluate=True)

        next_state, reward, done, _ = env.step(action)
        episode_reward += reward


        state = next_state


    print("----------------------------------------")
    print("Test Episodes: {}, Avg. Reward: {}".format(i_episode, round(episode_reward, 2)))
    print("----------------------------------------")
env.close()

