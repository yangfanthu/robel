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
import itertools


from modules import *
import utils

from tensorboardX import SummaryWriter


import pdb

def eval_policy(policy, env_name, broken_info = False, eval_episodes=3, real_robot = False, seed = 0):
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
            action = policy.select_action(np.array(state), evaluate=True)
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
    parser.add_argument("--adversary-start-timesteps", type=int, default=int(1e4))
    # parser.add_argument("--start-timesteps", type=int, default=int(256))
    # parser.add_argument("--adversary-start-timesteps", type=int, default=int(256))
    parser.add_argument("--max-timesteps", type=int, default=int(1e7))
    parser.add_argument("--eval-freq", type=int, default=20)
    parser.add_argument("--save-freq", type=int, default=5000)
    parser.add_argument("--record-freq", type=int, default=5000)
    # parser.add_argument("--eval-freq", type=int, default=1)
    # parser.add_argument("--save-freq", type=int, default=1)
    # parser.add_argument("--record-freq", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--buffer-max-size", type=int, default=int(1e6))
    parser.add_argument("--agent-training-episodes", type=int, default=int(2))
    parser.add_argument("--adversary-training-episodes", type=int,default=int(1))
    parser.add_argument("--restore-step", type=int, default=0)
    parser.add_argument("--broken-timesteps", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--broken-info", action='store_true', default=True,
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
    env.seed(args.seed)
    if not os.path.exists('./logs'):
        os.system('mkdir logs')
    if not os.path.exists('./saved_models'):
        os.system('mkdir saved_models')
    outdir = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    outdir = os.path.join('./saved_models', outdir)
    os.system('mkdir ' + outdir)
    with open(outdir+'/setting.txt','w') as f:
        f.writelines("fix the broken info bug\n")
        # f.writelines("don't fix the broken info bug\n")
        for each_arg, value in args.__dict__.items():
            f.writelines(each_arg + " : " + str(value)+"\n")
    writer = SummaryWriter(logdir=('logs/gac{}').format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_dim = env.reset().shape[0]
    if args.trim_state:
        state_dim -= 9
    original_state_dim = state_dim

    if args.broken_info:
        state_dim += 9
    action_dim = env.action_space.sample().shape[0]
    max_action = env.action_space.high[0]
    adversary = AdversarialDQN(state_dim=original_state_dim,
                               n_actions=action_dim,
                               device=device,
                               writer=writer,
                               buffer_max_size=args.buffer_max_size,
                               save_freq=args.save_freq,
                               record_freq=args.record_freq,
                               outdir=outdir)
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
    #     adversary.restore_model(args.restore_step)
    current_state = env.reset()
    if args.broken_info:
        joint_info = np.ones(9)
        current_state = np.concatenate((current_state, joint_info))
    def step(adversarial_action: int, current_state):
        "input obs obs is processed and its broken info must be all 1s"
        "return unprocessed next_state"
        if args.broken_info:
            current_state = np.concatenate((current_state, np.ones(9)))
            current_state[original_state_dim + adversarial_action] = 0
        # broken_timesteps = 1
        
        total_done = False
        reward_list = []
        for i in range(args.broken_timesteps):
            agent_action = agent.select_action(current_state, evaluate=True)
            agent_action[adversarial_action] = -0.6
            next_state, reward, done, info = env.step(agent_action)
            original_next_state = next_state
            if args.trim_state:
                next_state = utils.trim_state(next_state)
            if args.broken_info:
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
    episode = 0
    t = 0
    agent_t = 0
    adversary_t = 0
    
    done = False
    
    for i_episode in itertools.count(1):
        if t > args.max_timesteps:
            break
        for agent_episode in range(args.agent_training_episodes):
            done = False
            " the agent training loop"
            broken_joints = collections.deque(maxlen=1)
            current_state = env.reset()
            if args.trim_state:
                current_state = utils.trim_state(current_state)
            if args.broken_info:
                joint_info = np.ones(9)
                current_state = np.concatenate((current_state, joint_info))
            episode_steps = 0

            while not done:
                t += 1
                agent_t += 1
                
                
                "preprocess the state"
                if agent_t % args.broken_timesteps == 0 and agent_t > args.start_timesteps:
                    if args.broken_info:
                        adversary_action = adversary.select_action(current_state[:original_state_dim],'test')
                    else:
                        adversary_action = adversary.select_action(current_state,'test')
                    broken_joints.append(adversary_action[0])
                if args.broken_info:
                    for broken_one in broken_joints:
                        current_state[original_state_dim + broken_one] = 0
                
                if agent_t == args.start_timesteps:
                    print("start ddpg learning")
                if agent_t < args.start_timesteps:
                    original_action = env.action_space.sample()
                else:
                    original_action = agent.select_action(current_state, evaluate=False)
                action = copy.deepcopy(original_action)
                
                for broken_one in broken_joints:
                    action[broken_one] = args.broken_angle
                next_state, reward, done, info = env.step(action)
                episode_steps += 1
                mask = 1 if episode_steps == env._max_episode_steps else float(not done)
                if args.trim_state:
                    next_state = utils.trim_state(next_state)
                if args.broken_info:
                    next_state = np.concatenate((next_state, np.ones(9)))
                    for broken_one in broken_joints:
                        next_state[original_state_dim + broken_one] = 0
                # suc = info['score/success']
                agent.add_buffer(current_state, original_action, next_state, reward, mask)
                if agent_t > args.start_timesteps:
                    agent.update_parameters()
                current_state = next_state
                "fix the bug"
                current_state[original_state_dim:] = 1

        for adversary_episode in range(args.adversary_training_episodes):
            current_state = env.reset()
            if args.trim_state:
                current_state = utils.trim_state(current_state)
            "the adversary q training loop"
            done = False
            episode_steps = 0
            while not done:
                t += 1
                adversary_t += 1
                action = adversary.select_action(current_state,'train')
                next_state, reward, done, info = step(action[0],current_state)
                episode_steps += 1
                mask = 0 if episode_steps == env._max_episode_steps else float(done) # it's different, because my replay buffer adds done instead of not done
                if args.trim_state:
                    next_state = utils.trim_state(next_state)
                reward = -reward  # the adversary's target it to minimize the reward of the agent
                adversary.add_buffer(current_state, action, next_state, reward, mask)
                if adversary_t == args.adversary_start_timesteps:
                    print("start training the adversary!")
                if adversary_t > args.adversary_start_timesteps:
                    adversary.train()
                current_state = next_state

        if i_episode % args.eval_freq == 0:
            print("-------------------------------------------")
            print("steps:{:07d}".format(t))
            print("episode:{:07d}".format(i_episode))
            avg_reward = eval_policy(agent, 'DClawTurnFixed-v0', broken_info = args.broken_info)
            writer.add_scalar('/eval/avg_reward',avg_reward, i_episode)

