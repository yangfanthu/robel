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

def eval_policy(policy, env_name, eval_episodes=1, broken_info=False, real_robot = False, seed = 0):
    env_seed = 2 ** 32 - 1 - seed
    if real_robot:
        eval_env = gym.make(env_name, device_path='/dev/tty.usbserial-FT3WI485')
    else:
        eval_env = gym.make(env_name)
    eval_env.seed(env_seed)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        state = utils.trim_state(state)
        if broken_info:
            state = np.concatenate((state, np.ones(9)))
        while not done:
            action = policy.select_action(np.array(state), 'test')
            state, reward, done, _ = eval_env.step(action)
            state = utils.trim_state(state)
            if broken_info:
                state = np.concatenate((state, np.ones(9)))
            avg_reward += reward
        
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
    # parser.add_argument("--start-timesteps", type=int, default=int(1))
    parser.add_argument("--max-timesteps", type=int, default=int(1e7))
    # parser.add_argument("--eval-freq", type=int, default=1)
    # parser.add_argument("--save-freq", type=int, default=1)
    # parser.add_argument("--record-freq", type=int, default=1)    
    parser.add_argument("--eval-freq", type=int, default=5000)
    parser.add_argument("--save-freq", type=int, default=5000)
    parser.add_argument("--record-freq", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--buffer-max-size", type=int, default=int(1e6))
    parser.add_argument("--ddpg-training-steps", type=int, default=int(5000))
    # parser.add_argument("--ddpg-training-steps", type=int, default=int(2))
    parser.add_argument("--restore-step", type=int, default=0)
    parser.add_argument("--ddpg-hidden-size", type=int, default=512)
    parser.add_argument("--broken-info", action='store_true', default=True,
	                help="whether use broken joints indice as a part of state")
    parser.add_argument("--broken-info-recap", action='store_true', default=False,
						help='whether to use broken info again in actor module to reinforce the learning')
    parser.add_argument("--broken-angle", type=float, default=-0.6)
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
        f.writelines("model-based ddpg \n")
        f.writelines("for each training ddpg episode, only a specifc joint can be disabled, it cannot be changed to other ones in this episode\n")
        for each_arg, value in args.__dict__.items():
            f.writelines(each_arg + " : " + str(value)+"\n")
    writer = SummaryWriter(logdir=('logs/mbddpg{}').format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_dim = env.reset().shape[0] - 9
    original_state_dim = state_dim
    state_dim += 9 # add broken info
    action_dim = env.action_space.sample().shape[0]
    max_action = env.action_space.high[0]


    "advesarial agent code"
    ddpg = MBDDPG(state_dim=state_dim,
                  state_dim_2=9,
                  action_dim=action_dim,
                  device=device,
                  writer=writer,
                  buffer_max_size=args.buffer_max_size,
                  gamma=0.9,
                  batch_size=64,
                  max_action=max_action,
                  hidden_size=args.ddpg_hidden_size,
                  tau=5e-3,
                  variance=0.1,
                  save_freq=args.save_freq,
                  record_freq=args.record_freq,
                  outdir=outdir,
                  broken_info_recap=args.broken_info_recap,
                  broken_angle=args.broken_angle)
    ddpg.load_pretrained_model()
    if args.restore_step:
        print("restoring the model {}".format(args.restore_step))
        ddpg.restore_model_for_train(args.restore_step)
        # ddpg.index = 0
    current_state = env.reset()
    current_state = utils.trim_state(current_state)
    current_state = np.concatenate((current_state, np.ones(9)))
    episode = 0
    t = 0
    ddpg_t = 0
    adversary_t = 0
    minimal_index = 0
    def step(adversarial_action: int, ddpg_obs):
        current_state = ddpg_obs
        current_state[original_state_dim + adversarial_action] = 0
        broken_timesteps = 1
        
        total_done = False
        reward_list = []
        for i in range(broken_timesteps):
            ddpg_action = ddpg.select_action(current_state, 'test')
            ddpg_action[adversarial_action] = -0.6
            next_state, reward, done, info = env.step(ddpg_action)
            original_next_state = next_state
            next_state = utils.trim_state(next_state)
            next_state = np.concatenate((next_state, np.ones(9)))
            next_state[original_state_dim + adversarial_action] = 0
            reward_list.append(reward)
            if done:
                total_done = done
                break
            current_state = next_state
        avg_reward = np.array(reward_list).mean()
        return original_next_state, avg_reward, total_done, info

    while True:
        if t > args.max_timesteps:
            break
        
        " the ddpg training loop"
        broken_joints = collections.deque(maxlen=1)
        for i in range(args.ddpg_training_steps):
            t += 1
            ddpg_t += 1            
            if t % args.eval_freq == 0:
                print("-------------------------------------------")
                print("steps:{:07d}".format(t))
                print("episode:{:07d}".format(episode))
                avg_reward = eval_policy(ddpg, 'DClawTurnFixed-v0', broken_info=True)
                writer.add_scalar('/eval/avg_reward',avg_reward, t)
            
            if ddpg_t == args.start_timesteps:
                print("start ddpg learning")
            if ddpg_t < args.start_timesteps:
                original_action = env.action_space.sample()
            else:
                original_action = ddpg.select_action(current_state, 'train')
            action = copy.deepcopy(original_action)
            action[minimal_index] = - 0.6
            # action[adversary_action[0]] = 0
            next_state, reward, done, info = env.step(action)
            next_state = utils.trim_state(next_state)
            "broken info"
            next_state = np.concatenate((next_state, np.ones(9)))
            next_state[original_state_dim + minimal_index] = 0
            current_state[original_state_dim + minimal_index] = 0

            suc = info['score/success']
            ddpg.add_buffer(current_state, original_action, next_state, reward, done)
            ddpg.add_model_buffer(current_state, action, next_state, reward, done)
            if ddpg_t > args.start_timesteps:
                ddpg.train()
            current_state = next_state
            if done:
                broken_joints = collections.deque(maxlen=1)
                current_state = env.reset()
                current_state = utils.trim_state(current_state)
                "broken info"
                current_state = np.concatenate((current_state, np.ones(9)))
                episode += 1
        "the adversary q training loop"
        performance_list = []
        sum_reward = 0
        current_state = env.reset()
        current_state = utils.trim_state(current_state)
        current_state = np.concatenate((current_state, np.ones(9)))
        
        for i in range(action_dim):
            while True:
                next_state, reward, done, info = step(i, current_state)
                next_state = utils.trim_state(next_state)
                next_state = np.concatenate((next_state, np.ones(9)))
                next_state[original_state_dim + i] = 0
                sum_reward += reward
                current_state = next_state
                if done:
                    current_state = env.reset()
                    current_state = utils.trim_state(current_state)
                    current_state = np.concatenate((current_state, np.ones(9)))
                    performance_list.append(sum_reward)
                    sum_reward = 0
                    break
        performance_list = np.array(performance_list)
        minimal_index = np.where(performance_list == performance_list.min())
        minimal_index = minimal_index[0][0]
        current_state = env.reset()
        current_state = utils.trim_state(current_state)
        current_state = np.concatenate((current_state, np.ones(9)))
        


