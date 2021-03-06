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
    parser.add_argument("--broken-info", action='store_true', default=False,
                        help="whether use broken joints indice as a part of state")
    parser.add_argument("--broken-info-recap", action='store_true', default=False,
                        help='whether to use broken info again in actor module to reinforce the learning')
    parser.add_argument("--velocity-flag", action="store_true", default=True)
    args = parser.parse_args()
    if args.broken_info_recap:
        assert args.broken_info
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device('cpu')
    state_dim = env.reset().shape[0] - 9
    if args.velocity_flag:
        state_dim += 9
    original_state_dim = state_dim
    if args.broken_info:
        state_dim += 9
    action_dim = env.action_space.sample().shape[0]
    max_action = env.action_space.high[0]

    ddpg = DDPG(state_dim = state_dim,
    action_dim = action_dim,
    buffer_max_size=int(1e6),
    writer = None,
    max_action = max_action,
    device = device,
    hidden_size=512,
    broken_info_recap=args.broken_info_recap)
    ddpg.restore_model_for_test(4995000)
    adversary = AdversarialDQN(original_state_dim, action_dim, device, writer=None,buffer_max_size=int(1e6))
    # adversary.restore_model(2495000)
    current_state = env.reset()
    current_state = utils.trim_state(current_state)
    if args.velocity_flag:
        current_state = np.concatenate((current_state, np.zeros(9)))

    broken_joints = [] # 5 doesn't work

    if args.broken_info:
        current_state = np.concatenate((current_state, np.ones(9)))
        for broken_one in broken_joints:
            current_state[original_state_dim + broken_one] = 0
            if args.velocity_flag:
                current_state[9 + broken_one] = 0
    env.render()
    sum_reward = 0
    index = 0
    episode = 0
    env._max_episode_steps = 200
    with torch.no_grad():
        while True:
            if args.broken_info:
                adversary_action = adversary.select_action(current_state[:original_state_dim], 'test')
            else:
                adversary_action = adversary.select_action(current_state, 'test')
            # print(adversary_action)
            action = ddpg.select_action(current_state, 'test')
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
            # pdb.set_trace()
            next_state, reward, done, info = env.step(action)
            next_state = utils.trim_state(next_state)
            if args.velocity_flag:
                next_state = np.concatenate((next_state, next_state[:9] - current_state[:9]))
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
                current_state = utils.trim_state(current_state)
                if args.velocity_flag:
                    current_state = np.concatenate((current_state, np.zeros(9)))
                if args.broken_info:
                    current_state = np.concatenate((current_state, np.ones(9)))
                    for broken_one in broken_joints:
                        current_state[original_state_dim + broken_one] = 0
                print(sum_reward)
                # random_list.append(sum_reward)
                # print("episode: {}, avg: {}".format(episode, np.array(random_list).mean()))
                sum_reward = 0
            