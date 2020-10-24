import os
import sys
import copy
import datetime
import collections
import itertools
import robel
import gym
import torch
import torch.nn as nn
import gym
import numpy as np
import argparse


from modules import *
import utils

from tensorboardX import SummaryWriter


import pdb

def eval_policy(policy, env_name, broken_info = False, eval_episodes=2, real_robot = False, seed = 0):
    env_seed = 2 ** 32 - 1 - seed
    if real_robot:
        eval_env = gym.make(env_name, device_path='/dev/tty.usbserial-FT3WI485')
    else:
        eval_env = gym.make(env_name)
    eval_env.seed(env_seed)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        if args.trim_state:
            state = utils.trim_state(state)
        if broken_info:
            state = np.concatenate((state, np.ones(9)))
        while not done:
            action = policy.select_action(np.array(state), True)
            state, reward, done, _ = eval_env.step(action)
            if args.trim_state:
                state = utils.trim_state(state)
            avg_reward += reward
            if broken_info:
                state = np.concatenate((state, np.ones(9)))
        
    avg_reward /= eval_episodes

    print("---------------------------------------")
    print("Evaluation over {} episodes: {:.3f}".format(eval_episodes, avg_reward))
    print("---------------------------------------")
    return avg_reward

env = gym.make('DKittyWalkFixed-v0')
# env = gym.make('DClawTurnFixed-v0', device_path='/dev/tty.usbserial-FT3WI485')
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-timesteps", type=int, default=int(1e4))
    # parser.add_argument("--start-timesteps", type=int, default=int(4))
    parser.add_argument("--max-timesteps", type=int, default=int(1e7))
    parser.add_argument("--eval-freq", type=int, default=1000)
    parser.add_argument("--save-freq", type=int, default=5000)
    parser.add_argument("--record-freq", type=int, default=5000)
    # parser.add_argument("--eval-freq", type=int, default=1)
    # parser.add_argument("--save-freq", type=int, default=1)
    # parser.add_argument("--record-freq", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--buffer-max-size", type=int, default=int(1e6))
    parser.add_argument("--agent-training-steps", type=int, default=int(5000))
    # parser.add_argument("--agent-training-steps", type=int, default=int(2))
    parser.add_argument("--restore-step", type=int, default=0)
    parser.add_argument("--broken-timesteps", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--broken-info", action='store_true', default=False,
                        help="whether use broken joints indice as a part of state")
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
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, metavar='G',
                        help='Automaically adjust α (default: False)')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                        help='learning rate (default: 0.0003)')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='batch size (default: 256)')
    parser.add_argument('--trim-state', action="store_true", default=False)
    args = parser.parse_args()
    env.seed(args.seed)
    if not os.path.exists('./logs'):
        os.system('mkdir logs')
    if not os.path.exists('./saved_models'):
        os.system('mkdir saved_models')
    outdir = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    outdir = os.path.join('./saved_models', outdir)
    os.system('mkdir ' + outdir)
    with open(outdir+'/setting.txt','w') as f:
        for each_arg, value in args.__dict__.items():
            f.writelines("pure sac\n")
            f.writelines(each_arg + " : " + str(value)+"\n")
    writer = SummaryWriter(logdir=('logs/gac{}').format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_dim = env.reset().shape[0]
    if args.trim_state:
        state_dim -= 9
    original_state_dim = state_dim

    action_dim = env.action_space.sample().shape[0]
    max_action = env.action_space.high[0]
    agent = SAC(num_inputs=state_dim,
                action_space=env.action_space,
                args=args,
                writer=writer,
                outdir=outdir,
                device=device)
 
    total_numsteps = 0
    for i_episode in itertools.count(1):
        episode_reward = 0
        episode_steps = 0
        done = False
        state = env.reset()
        if args.trim_state:
            state = utils.trim_state(state)

        while not done:
            if args.start_timesteps > total_numsteps:
                action = env.action_space.sample()  # Sample random action
            else:
                action = agent.select_action(state)  # Sample action from policy

            if len(agent.replay_buffer) > args.batch_size:
                # Number of updates per step in environment
                for i in range(1):
                    # Update parameters of all the networks
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters()


            next_state, reward, done, _ = env.step(action) # Step
            if args.trim_state:
                next_state = utils.trim_state(next_state)
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward

            # Ignore the "done" signal if it comes from hitting the time horizon.
            # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
            mask = 1 if episode_steps == env._max_episode_steps else float(not done)
            agent.replay_buffer.push(state, action, next_state, reward, mask) # Append transition to memory

            state = next_state
        writer.add_scalar('reward/train', episode_reward, i_episode)
        if i_episode % 20 == 0:
            print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))

        if i_episode % 10 == 0 and args.eval is True:

            "original evaluate code"
            avg_reward = 0.
            episodes = 10
            for _  in range(episodes):
                state = env.reset()
                if args.trim_state:
                    state = utils.trim_state(state)
                episode_reward = 0
                done = False
                while not done:
                    action = agent.select_action(state, evaluate=True)

                    next_state, reward, done, _ = env.step(action)
                    if args.trim_state:
                        next_state = utils.trim_state(next_state)
                    episode_reward += reward


                    state = next_state
                avg_reward += episode_reward
            avg_reward /= episodes


            writer.add_scalar('avg_reward/test', avg_reward, i_episode)
            print("----------------------------------------")
            print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
            print("----------------------------------------")

    env.close()



