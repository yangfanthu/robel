import robel
import gym
import torch
import torch.nn as nn
import gym
import numpy as np
import os
import sys
import copy
import datetime
import argparse
import collections


from modules import *
import utils

from tensorboardX import SummaryWriter


import pdb

def eval_policy(policy, env_name, eval_episodes=1, broken_info=False, real_robot = False, seed = 0, trim_state_bool=False, velocity_flag = False):
    env_seed = 2 ** 32 - 1 - seed
    if real_robot:
        eval_env = gym.make(env_name, device_path='/dev/tty.usbserial-FT3WI485')
    else:
        eval_env = gym.make(env_name)
    eval_env.seed(env_seed)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        if trim_state_bool:
            state = trim_state(state)
        if broken_info:
            state = np.concatenate((state, np.ones(9)))
        if velocity_flag:
            state = np.concatenate((state, np.zeros(9)))
        while not done:
            action = policy.select_action(np.array(state), 'test')
            prev_state = state
            state, reward, done, _ = eval_env.step(action)
            if trim_state_bool:
                state = trim_state(state)
            if broken_info:
                state = np.concatenate((state, np.ones(9)))
            if velocity_flag:
                state = np.concatenate((state, state[:9] - prev_state[:9]))
            avg_reward += reward
        
    avg_reward /= eval_episodes

    print("---------------------------------------")
    print("Evaluation over {} episodes: {:.3f}".format(eval_episodes, avg_reward))
    print("---------------------------------------")
    return avg_reward
def trim_state(current_state):
    joint = current_state[:9]
    cossin = current_state[9:11]
    command = current_state[11:20]
    other = current_state[20]
    output_state = np.concatenate((joint,cossin, np.array([other])), axis=0)
    return output_state
env = gym.make('DClawTurnFixed-v0')
# env = gym.make('DClawTurnFixed-v0', device_path='/dev/tty.usbserial-FT3WI485')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-timesteps", type=int, default=int(1e4))
    # parser.add_argument("--start-timesteps", type=int, default=int(1))
    parser.add_argument("--max-timesteps", type=int, default=int(1e7))
    # parser.add_argument("--eval-freq", type=int, default=1)
    parser.add_argument("--eval-freq", type=int, default=5000)
    parser.add_argument("--save-freq", type=int, default=5000)
    parser.add_argument("--record-freq", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--buffer-max-size", type=int, default=int(1e6))
    parser.add_argument("--restore-step", type=int, default=0)
    parser.add_argument("--ddpg-hidden-size", type=int, default=512)
    parser.add_argument("--std", type=float, default=0.1)
    parser.add_argument("--trim-state", action="store_true", default=True)
    parser.add_argument("--velocity-flag", action="store_true", default=True)
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
        f.writelines("pure ddpg")
        for each_arg, value in args.__dict__.items():
            f.writelines(each_arg + " : " + str(value)+"\n")
    writer = SummaryWriter(logdir=('logs/{}').format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.trim_state:
        state_dim = env.reset().shape[0] - 9
    else:
        state_dim = env.reset().shape[0]
    if args.velocity_flag:
        state_dim = state_dim + 9
    original_state_dim = state_dim
    action_dim = env.action_space.sample().shape[0]
    max_action = env.action_space.high[0]


    "advesarial agent code"
    ddpg = DDPG(state_dim=state_dim,
                action_dim=action_dim,
                device=device,
                writer=writer,
                buffer_max_size=args.buffer_max_size,
                gamma=0.9,
                batch_size=64,
                max_action=max_action,
                hidden_size=args.ddpg_hidden_size,
                tau=5e-3,
                variance=args.std,
                save_freq=args.save_freq,
                record_freq=args.record_freq,
                outdir=outdir)
    if args.restore_step:
        print("restoring the model {}".format(args.restore_step))
        ddpg.restore_model_for_train(args.restore_step)
    current_state = env.reset()
    if args.trim_state:
        current_state = trim_state(current_state)
    if args.velocity_flag:
        current_state = np.concatenate((current_state, np.zeros(9)), axis=-1)
    episode = 0
    t = 0
    adversary_t = 0
    minimal_index = 0
    while True:
        if t > args.max_timesteps:
            break
        
        " the ddpg training loop"
        t += 1            
        if t % args.eval_freq == 0:
            print("-------------------------------------------")
            print("steps:{:07d}".format(t))
            print("episode:{:07d}".format(episode))
            avg_reward = eval_policy(ddpg, 'DClawTurnFixed-v0', trim_state_bool=args.trim_state, velocity_flag=args.velocity_flag)
            writer.add_scalar('/eval/avg_reward',avg_reward, t)
        
        if t == args.start_timesteps:
            print("start ddpg learning")
        if t < args.start_timesteps:
            original_action = env.action_space.sample()
        else:
            original_action = ddpg.select_action(current_state, 'train')
        # action[adversary_action[0]] = 0
        next_state, reward, done, info = env.step(original_action)
        if args.trim_state:
            next_state = trim_state(next_state)
        if args.velocity_flag:
            next_state = np.concatenate((next_state, next_state[:9] - current_state[:9]), axis=-1)
        suc = info['score/success']
        ddpg.add_buffer(current_state, original_action, next_state, reward, done)
        if t > args.start_timesteps:
            ddpg.train()
        current_state = next_state
        if done:
            current_state = env.reset()
            if args.trim_state:
                current_state = trim_state(current_state)
            if args.velocity_flag:
                current_state = np.concatenate((current_state, np.zeros(9)), axis=-1)
            episode += 1
