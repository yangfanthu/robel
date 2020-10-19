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
    parser.add_argument("--broken-info", action='store_true', default=True,
                        help="whether use broken joints indice as a part of state")
    parser.add_argument("--broken-info-recap", action='store_true', default=False,
                        help='whether to use broken info again in actor module to reinforce the learning')
    parser.add_argument("--save-freq", type=int, default=5000)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--buffer-max-size", type=int, default=int(1e6))
    parser.add_argument("--agent-training-episodes", type=int, default=int(2))
    parser.add_argument("--adversary-training-episodes", type=int,default=int(1))
    parser.add_argument("--restore-step", type=int, default=0)
    parser.add_argument("--broken-timesteps", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--broken-angle", type=float, default=-0.6)
    parser.add_argument("--std", type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--env-name', default='DClawTurnFixed-v0',
                        help='Mujoco Gym environment (default: HalfCheetah-v2)')
    parser.add_argument('--policy', default="Gaussian",
                        help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--eval', type=bool, default=True,
                        help='Evaluates a policy a policy every 10 episode (default: True)')
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                        help='Temperature parameter α determines the relative importance of the entropy\
                                term against the reward (default: 0.2)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                        help='Automaically adjust α (default: False)')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                        help='learning rate (default: 0.0003)')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='batch size (default: 256)')
    parser.add_argument('--trim-state', action="store_true", default=True)
    args = parser.parse_args()
    if args.broken_info_recap:
        assert args.broken_info
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device('cpu')
    state_dim = env.reset().shape[0]
    if args.trim_state:
        state_dim -= 9
    original_state_dim = state_dim
    if args.broken_info:
        state_dim += 9
    action_dim = env.action_space.sample().shape[0]
    max_action = env.action_space.high[0]

    # agent = DDPG(state_dim = state_dim,
    # action_dim = action_dim,
    # buffer_max_size=int(1e6),
    # writer = None,
    # max_action = max_action,
    # device = device,
    # hidden_size=512,
    # broken_info_recap=args.broken_info_recap)
    agent = SAC(num_inputs=state_dim,
                action_space=env.action_space,
                args=args,
                writer=None,
                outdir=None,
                device=device)
    agent.restore_model_for_test(240000)
    adversary = AdversarialDQN(original_state_dim, action_dim, device, writer=None,buffer_max_size=int(1e6))
    # adversary.restore_model(2495000)
    current_state = env.reset()
    if args.trim_state:
        current_state = utils.trim_state(current_state)

    broken_joints = [] # 5 doesn't work

    if args.broken_info:
        current_state = np.concatenate((current_state, np.ones(9)))
        for broken_one in broken_joints:
            current_state[original_state_dim + broken_one] = 0
    env.render()
    sum_reward = 0
    index = 0
    episode = 0
    #env._max_episode_steps = 200
    with torch.no_grad():
        while True:
            if args.broken_info:
                adversary_action = adversary.select_action(current_state[:original_state_dim], 'test')
            else:
                adversary_action = adversary.select_action(current_state, 'test')
            # print(adversary_action)
            action = agent.select_action(current_state, 'test')
            index += 1
            for broken_one in broken_joints:
                action[broken_one] = -0.6
            # action = np.ones(9) * -0.6
            # action[4] = -0.6  # this case doesn't work, also 6,7
            # action[1] = -0.6

            # action[random.randint(0,8)] = 0
            # print(adversary_action)
            # action[adversary_action[0]] = -0.6
            # action[random.randint(0,8)] = 0
            # joint = current_state[:9]
            # cossin = current_state[9:11]
            # command = current_state[11:20]
            # other = current_state[20]
            # print(joint)
            # print(other)
            next_state, reward, done, info = env.step(action)
            if args.trim_state:
                next_state = utils.trim_state(next_state)
            if args.broken_info:
                next_state = np.concatenate((next_state, np.ones(9)))
                for broken_one in broken_joints:
                    current_state[original_state_dim + broken_one] = 0
                    next_state[original_state_dim + broken_one] = 0
            sum_reward += reward 
            # print(sum_reward)
            suc = info['score/success']
            
            current_state = next_state
            if done:
                episode += 1
                current_state = env.reset()
                if args.trim_state:
                    current_state = utils.trim_state(current_state)
                if args.broken_info:
                    current_state = np.concatenate((current_state, np.ones(9)))
                
                    for broken_one in broken_joints:
                        current_state[original_state_dim + broken_one] = 0
                print(sum_reward)
                # random_list.append(sum_reward)
                # print("episode: {}, avg: {}".format(episode, np.array(random_list).mean()))
                sum_reward = 0
            