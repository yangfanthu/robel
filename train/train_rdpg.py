import os
import datetime
import argparse
import numpy as np
import torch
# import torch.nn as nn
import gym
import utils
import robel

from modules import *
from tensorboardX import SummaryWriter

import pdb

env = gym.make('DClawTurnFixed-v0')
max_episode_steps = env._max_episode_steps
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# env = gym.make('DClawTurnFixed-v0', device_path='/dev/tty.usbserial-FT3WI485')

def eval_policy(policy, env_name, eval_episodes=10, real_robot = False, seed = 0):
    env_seed = 2 ** 32 - 1 - seed
    state_dim = env.reset().shape[0]
    if real_robot:
        eval_env = gym.make(env_name, device_path='/dev/tty.usbserial-FT3WI485')
    else:
        eval_env = gym.make(env_name)
    eval_env.seed(env_seed)

    avg_reward = 0.
    for _ in range(eval_episodes):
        current_state_buffer = np.zeros((max_episode_steps, state_dim))
        state, done = eval_env.reset(), False
        policy.reset()
        index = 0
        current_state_buffer[0] = state
        while not done:
            action = policy.select_action(current_state_buffer, 'test')
            state, reward, done, _ = eval_env.step(action)
            index += 1
            if index < max_episode_steps:
                current_state_buffer[index] = state
            avg_reward += reward
        
    avg_reward /= eval_episodes

    print("---------------------------------------")
    print("Evaluation over {} episodes: {:.3f}".format(eval_episodes, avg_reward))
    print("---------------------------------------")
    return avg_reward

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-timesteps", type=int, default=int(1e4))
    # parser.add_argument("--start-timesteps", type=int, default=int(2))
    parser.add_argument("--max-timesteps", type=int, default=int(5e6))
    parser.add_argument("--eval-freq", type=int, default=5000)
    parser.add_argument("--save-freq", type=int, default=5000)
    parser.add_argument("--record-freq", type=int, default=5000)
    # parser.add_argument("--eval-freq", type=int, default=1)
    # parser.add_argument("--save-freq", type=int, default=1)
    # parser.add_argument("--record-freq", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--buffer-max-size", type=int, default=int(2e4))
    parser.add_argument("--ddpg-training-steps", type=int, default=int(5000))
    # parser.add_argument("--ddpg-training-steps", type=int, default=int(3))
    parser.add_argument("--restore-step", type=int, default=0)
    parser.add_argument("--ddpg-hidden-size", type=int, default=512)
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
        f.writelines("RDPG agent")
        f.writelines("for each training ddpg episode, only a specifc joint can be disabled, it cannot be changed to other ones in this episode")
        for each_arg, value in args.__dict__.items():
            f.writelines(each_arg + " : " + str(value)+"\n")
    writer = SummaryWriter(logdir=('logs/{}').format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_dim = env.reset().shape[0]
    action_dim = env.action_space.sample().shape[0]
    max_action = env.action_space.high[0]


    "advesarial agent code"
    rdpg_agent = RDPG(state_dim=state_dim,
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
                      max_episode_timesteps=max_episode_steps)
    if args.restore_step:
        print("restoring the model {}".format(args.restore_step))
        rdpg_agent.restore_model(args.restore_step)
    current_state = env.reset()
    episode = 0
    t = 0
    ddpg_t = 0
    adversary_t = 0
    minimal_index = 0
    current_state_buffer = np.zeros((max_episode_steps, state_dim))
    action_buffer = np.zeros((max_episode_steps, action_dim))
    next_state_buffer = np.zeros((max_episode_steps, state_dim))
    reward_buffer = np.zeros((max_episode_steps, 1))
    done_buffer = np.ones((max_episode_steps, 1))
    current_state_buffer[0] = current_state
    episode_t = 0
    while True:
        if t > args.max_timesteps:
            break
        " the ddpg training loop"
        episode_t = 0
        for i in range(args.ddpg_training_steps):
            t += 1
            ddpg_t += 1
            if t % args.eval_freq == 0:
                print("-------------------------------------------")
                print("steps:{:07d}".format(t))
                print("episode:{:07d}".format(episode))
                avg_reward = eval_policy(rdpg_agent, 'DClawTurnFixed-v0')
                writer.add_scalar('/eval/avg_reward',avg_reward, t)
            
            if ddpg_t == args.start_timesteps:
                print("start ddpg learning")
            if ddpg_t <= args.start_timesteps:
                original_action = env.action_space.sample()
            else:
                original_action = rdpg_agent.select_action(current_state_buffer, 'train')
            action = copy.deepcopy(original_action)
            action[minimal_index] = - 0.6
            # action[adversary_action[0]] = 0
            next_state, reward, done, info = env.step(action)
            suc = info['score/success']
            # current_state_buffer[episode_t] = current_state
            if episode_t < max_episode_steps:
                action_buffer[episode_t] = original_action
                next_state_buffer[episode_t] = next_state
                reward_buffer[episode_t] = reward
                done_buffer[episode_t] = done
                if episode_t + 1 < max_episode_steps:
                    current_state_buffer[episode_t + 1] = next_state
            episode_t += 1
            if ddpg_t == args.start_timesteps:
                done = True  
                # when start using rdpg agent to roll out, we reset all the parameters
            # rdpg_agent.add_buffer(current_state, original_action, next_state, reward, done, episode_t)
            if ddpg_t > args.start_timesteps:
                rdpg_agent.train()
            if done:
                current_state = env.reset()
                rdpg_agent.add_buffer(current_state_buffer,
                                      action_buffer,
                                      next_state_buffer,
                                      reward_buffer,
                                      done_buffer,
                                      episode_t)
                #episode t point to the next empty space index
                current_state_buffer = np.zeros((max_episode_steps, state_dim))
                action_buffer = np.zeros((max_episode_steps, action_dim))
                next_state_buffer = np.zeros((max_episode_steps, state_dim))
                reward_buffer = np.zeros((max_episode_steps, 1))
                done_buffer = np.ones((max_episode_steps, 1))
                current_state_buffer[0] = current_state
                rdpg_agent.reset()
                episode_t = 0
                episode += 1
            # current_state = env.reset()
        
        "the adversary q training loop"
        performance_list = []
        for i in range(action_dim):
            reward_list = []
            sum_reward = 0
            current_state = env.reset()
            episode_t = 0
            rdpg_agent.reset()
            current_state_buffer = np.zeros((max_episode_steps, state_dim))
            action_buffer = np.zeros((max_episode_steps, action_dim))
            next_state_buffer = np.zeros((max_episode_steps, state_dim))
            reward_buffer = np.zeros((max_episode_steps, 1))
            done_buffer = np.ones((max_episode_steps, 1))
            current_state_buffer[0] = current_state
            while True:
                original_action = rdpg_agent.select_action(current_state_buffer, 'test')
                action = copy.deepcopy(original_action)
                action[i] = -0.6
                next_state, reward, done, info = env.step(action)
                reward_list.append(reward)
                # sum_reward += reward
                current_state = next_state
                # current_state_buffer[episode_t] = current_state
                if episode_t < max_episode_steps:
                    action_buffer[episode_t] = original_action
                    next_state_buffer[episode_t] = next_state
                    reward_buffer[episode_t] = reward
                    done_buffer[episode_t] = done
                    if episode_t + 1 < max_episode_steps:
                        current_state_buffer[episode_t + 1] = next_state
                if done:
                    current_state = env.reset()
                    avg_reward = np.array(reward_list).mean()
                    performance_list.append(avg_reward)
                    current_state_buffer = np.zeros((max_episode_steps, state_dim))
                    action_buffer = np.zeros((max_episode_steps, action_dim))
                    next_state_buffer = np.zeros((max_episode_steps, state_dim))
                    reward_buffer = np.zeros((max_episode_steps, 1))
                    done_buffer = np.ones((max_episode_steps, 1))
                    current_state_buffer[0] = current_state
                    rdpg_agent.reset()
                    break
        performance_list = np.array(performance_list)
        minimal_index = np.where(performance_list == performance_list.min())
