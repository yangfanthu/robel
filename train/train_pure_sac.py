import os
import sys
import copy
import datetime
import collections
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

def eval_policy(policy, env_name, broken_info = False, eval_episodes=5, real_robot = False, seed = 0):
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
            action = policy.select_action(np.array(state), 'test')
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

env = gym.make('DClawTurnFixed-v0')
# env = gym.make('DClawTurnFixed-v0', device_path='/dev/tty.usbserial-FT3WI485')
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-timesteps", type=int, default=int(1e4))
    # parser.add_argument("--start-timesteps", type=int, default=int(4))
    parser.add_argument("--max-timesteps", type=int, default=int(1e7))
    parser.add_argument("--eval-freq", type=int, default=5000)
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
    parser.add_argument('--trim-state', action="store_true", default=True)
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
    "advesarial agent code"
    # if args.restore_step:
    #     print("restoring the model {}".format(args.restore_step))
    #     ddpg.restore_model_for_train(args.restore_step)
    #     ddpg.index = 0
    current_state = env.reset()
    episode = 0
    t = 0
    agent_t = 0
    episode_steps = 0
    
    while True:
        if t > args.max_timesteps:
            break
        
        " the agent training loop"
        current_state = env.reset()
        if args.trim_state:
            current_state = utils.trim_state(current_state)
        for i in range(args.agent_training_steps):
            t += 1
            agent_t += 1
            
            if t % args.eval_freq == 0:
                print("-------------------------------------------")
                print("steps:{:07d}".format(t))
                print("episode:{:07d}".format(episode))
                avg_reward = eval_policy(agent, 'DClawTurnFixed-v0', broken_info = args.broken_info)
                writer.add_scalar('/eval/avg_reward',avg_reward, t)
            

            if agent_t == args.start_timesteps:
                print("start ddpg learning")
            if agent_t < args.start_timesteps:
                original_action = env.action_space.sample()
            else:
                original_action = agent.select_action(current_state)
            action = copy.deepcopy(original_action)
            
            next_state, reward, done, info = env.step(action)
            if args.trim_state:
                next_state = utils.trim_state(next_state)
            mask = 1 if episode_steps == env._max_episode_steps else float(not done)
            episode_steps += 1
            suc = info['score/success']
            agent.add_buffer(current_state, original_action, next_state, reward, done)
            if agent_t > args.start_timesteps:
                agent.update_parameters()
            current_state = next_state
            "fix the bug"
            current_state[original_state_dim:] = 1
            if done:
                episode_steps = 0
                current_state = env.reset()
                if args.trim_state:
                    current_state = utils.trim_state(current_state)
                episode += 1


