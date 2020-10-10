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
        if broken_info:
            state = np.concatenate((state, np.ones(9)))
        while not done:
            action = policy.select_action(np.array(state), 'test')
            state, reward, done, _ = eval_env.step(action)
            if broken_info:
                state = np.concatenate((state, np.ones(9)))
            avg_reward += reward
        
    avg_reward /= eval_episodes

    print("---------------------------------------")
    print("Evaluation over {} episodes: {:.3f}".format(eval_episodes, avg_reward))
    print("---------------------------------------")
    return avg_reward

base_env = gym.make('DClawTurnFixed-v0')
# env = gym.make('DClawTurnFixed-v0', device_path='/dev/tty.usbserial-FT3WI485')

class AdversarialEnv(object):
    def __init__(self,
                 ddpg_action_dim,
                 ddpg_state_dim,
                 ddpg_buffer_max_size,
                 writer,
                 ddpg_gamma,
                 ddpg_hidden_size,
                 ddpg_save_freq,
                 ddpg_record_freq,
                 ddpg_batch_size,
                 ddpg_max_action,
                 ddpg_tau,
                 ddpg_variance,
                 device,
                 broken_timesteps,
                 broken_info = False,
                 env_name='DClawTurnFixed-v0',
                 real_robot=False,
                 outdir = None,
                 broken_info_recap = False):
        self.ddpg_action_dim = ddpg_action_dim
        self.ddpg_state_dim = ddpg_state_dim
        self.ddpg_buffer_max_size = ddpg_buffer_max_size
        self.writer = writer
        self.ddpg_gamma = ddpg_gamma
        self.ddpg_hidden_size = ddpg_hidden_size
        self.ddpg_save_freq = ddpg_save_freq
        self.device = device
        self.env_name = env_name 
        self.real_robot = real_robot
        self.ddpg = DDPG(state_dim=ddpg_state_dim,
                         action_dim=ddpg_action_dim,
                         device=device,
                         writer=writer,
                         buffer_max_size=ddpg_buffer_max_size,
                         gamma=ddpg_gamma,
                         save_freq=ddpg_save_freq,
                         record_freq=ddpg_record_freq,
                         outdir = outdir,
                         hidden_size=ddpg_hidden_size,
                         broken_info_recap=broken_info_recap)
        self.broken_timesteps = broken_timesteps
        if real_robot:
            self.base_env = gym.make(env_name, device_path='/dev/tty.usbserial-FT3WI485')
        else:
            self.base_env = gym.make(env_name)
        self.action_space = self.base_env.action_space
        self.outdir = outdir
        self.broken_info = broken_info
    def step(self, adversarial_action: int, ddpg_obs):
        current_state = ddpg_obs
        original_state_dim = current_state.shape[0]
        if self.broken_info:
            current_state = np.concatenate((current_state, np.ones(9)))
            current_state[original_state_dim + adversarial_action] = 0
        total_done = False
        reward_list = []
        for i in range(self.broken_timesteps):
            ddpg_action = self.ddpg.select_action(current_state, 'test')
            ddpg_action[adversarial_action] = -0.6
            next_state, reward, done, info = self.base_env.step(ddpg_action)
            original_next_state = next_state
            if self.broken_info:
                joint_info = np.ones(9)
                joint_info[adversarial_action] = 0
                next_state = np.concatenate((next_state, joint_info))
            reward_list.append(reward)
            if done:
                total_done = done
                break
            current_state = next_state
        avg_reward = np.array(reward_list).mean()
        return original_next_state, avg_reward, total_done, info
        # ddpg_action = self.ddpg.select_action(ddpg_obs, 'test')
        # ddpg_action[advesarial_action] = 0.
        # next_state, reward, done, info  = self.base_env.step(ddpg_action)
        # return next_state, reward, done, info
    def seed(self, input_seed):
        self.base_env.seed(input_seed)
    def reset(self):
        return self.base_env.reset()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-timesteps", type=int, default=int(1e4))
    parser.add_argument("--max-timesteps", type=int, default=int(1e7))
    parser.add_argument("--eval-freq", type=int, default=5000)
    parser.add_argument("--save-freq", type=int, default=5000)
    parser.add_argument("--record-freq", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--buffer-max-size", type=int, default=int(1e6))
    parser.add_argument("--ddpg-training-steps", type=int, default=int(5000))
    parser.add_argument("--restore-step", type=int, default=0)
    parser.add_argument("--ddpg-hidden-size", type=int, default=512)
    parser.add_argument("--broken-info", action='store_true', default=True,
	                help="whether use broken joints indice as a part of state")
    parser.add_argument("--broken-info-recap", action='store_true', default=False,
						help='whether to use broken info again in actor module to reinforce the learning')
    args = parser.parse_args()
    if args.broken_info_recap:
        assert args.broken_info
    base_env.seed(args.seed)
    if not os.path.exists('./logs'):
        os.system('mkdir logs')
    if not os.path.exists('./saved_models'):
        os.system('mkdir saved_models')
    outdir = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    outdir = os.path.join('./saved_models', outdir)
    os.system('mkdir ' + outdir)
    with open(outdir+'/setting.txt','w') as f:
        f.writelines("for each training ddpg episode, only a specifc joint can be disabled, it cannot be changed to other ones in this episode")
        for each_arg, value in args.__dict__.items():
            f.writelines(each_arg + " : " + str(value)+"\n")
    writer = SummaryWriter(logdir=('logs/{}').format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_dim = base_env.reset().shape[0]
    original_state_dim = state_dim
    if args.broken_info:
        state_dim += 9
    action_dim = base_env.action_space.sample().shape[0]
    max_action = base_env.action_space.high[0]


    "advesarial agent code"
    adversarial_env = AdversarialEnv(ddpg_action_dim = action_dim,
                                     ddpg_state_dim=state_dim,
                                     ddpg_buffer_max_size=args.buffer_max_size,
                                     writer=writer,
                                     device=device,
                                     ddpg_gamma=0.9,
                                     ddpg_hidden_size=args.ddpg_hidden_size,
                                     ddpg_save_freq=args.save_freq,
                                     ddpg_record_freq=args.record_freq,
                                     ddpg_batch_size=64,
                                     ddpg_max_action=max_action,
                                     ddpg_tau=5e-3,
                                     ddpg_variance=0.1,
                                     broken_timesteps=1,
                                     broken_info=args.broken_info,
                                     outdir = outdir,
                                     broken_info_recap=args.broken_info_recap)
    if args.restore_step:
        print("restoring the model {}".format(args.restore_step))
        adversarial_env.ddpg.restore_model_for_train(args.restore_step)
        # adversarial_env.ddpg.index = 0
    current_state = adversarial_env.reset()
    if args.broken_info:
        current_state = np.concatenate((current_state, np.ones(9)))
    episode = 0
    t = 0
    ddpg_t = 0
    adversary_t = 0
    minimal_index = 0
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
                avg_reward = eval_policy(adversarial_env.ddpg, 'DClawTurnFixed-v0', broken_info=args.broken_info)
                writer.add_scalar('/eval/avg_reward',avg_reward, t)
            
            if ddpg_t == args.start_timesteps:
                print("start ddpg learning")
            if ddpg_t < args.start_timesteps:
                original_action = adversarial_env.action_space.sample()
            else:
                original_action = adversarial_env.ddpg.select_action(current_state, 'train')
            action = copy.deepcopy(original_action)
            action[minimal_index] = - 0.6
            # action[adversary_action[0]] = 0
            next_state, reward, done, info = adversarial_env.base_env.step(action)
            if args.broken_info:
                next_state = np.concatenate((next_state, np.ones(9)))
                next_state[original_state_dim + minimal_index] = 0
                current_state[original_state_dim + minimal_index] = 0
            suc = info['score/success']
            adversarial_env.ddpg.add_buffer(current_state, original_action, next_state,reward,done)
            if ddpg_t > args.start_timesteps:
                adversarial_env.ddpg.train()
            current_state = next_state
            if done:
                broken_joints = collections.deque(maxlen=1)
                current_state = adversarial_env.reset()
                if args.broken_info:
                    current_state = np.concatenate((current_state, np.ones(9)))
                episode += 1
        "the adversary q training loop"
        performance_list = []
        sum_reward = 0
        current_state = adversarial_env.reset()
        for i in range(action_dim):
            while True:
                next_state, reward, done, info = adversarial_env.step(i, current_state)
                sum_reward += reward
                current_state = next_state
                if done:
                    current_state = adversarial_env.reset()
                    performance_list.append(sum_reward)
                    sum_reward = 0
                    break
        performance_list = np.array(performance_list)
        minimal_index = np.where(performance_list == performance_list.min())
        minimal_index = minimal_index[0][0]
        current_state = adversarial_env.reset()
        if args.broken_info:
            current_state = np.concatenate((current_state, np.ones(9)))


