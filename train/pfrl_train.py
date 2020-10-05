"using the same algorithm as train.py, but implemented with pfrl framework"
import argparse
import logging
import sys

import robel
import gym
import gym.wrappers
import numpy as np
import torch
from torch import nn

import pfrl
from pfrl.agents.ddpg import DDPG
from pfrl.agents.dqn import DQN
from pfrl import experiments
from pfrl import explorers
from pfrl import utils
from pfrl import replay_buffers
from pfrl.nn import ConcatObsAndAction
from pfrl.nn import BoundByTanh
from pfrl.policies import DeterministicHead
from pfrl.experiments.evaluator import Evaluator
from pfrl.experiments.evaluator import save_agent

from pfrl.q_functions import DiscreteActionValueHead
import pdb


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        type=str,
        default="results",
        help=(
            "Directory path to save output files."
            " If it does not exist, it will be created."
        ),
    )
    parser.add_argument(
        "--env",
        type=str,
        default="'DClawTurnFixed-v0'",
        help="OpenAI Gym MuJoCo env to perform algorithm on.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed [0, 2 ** 32)")
    parser.add_argument(
        "--gpu", type=int, default=-1, help="GPU to use, set to -1 if no GPU."
    )
    parser.add_argument(
        "--load", type=str, default="", help="Directory to load agent from."
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10 ** 6,
        help="Total number of timesteps to train the agent.",
    )
    parser.add_argument(
        "--eval-n-runs",
        type=int,
        default=10,
        help="Number of episodes run for each evaluation.",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=5000,
        help="Interval in timesteps between evaluations.",
    )
    parser.add_argument(
        "--replay-start-size",
        type=int,
        default=10000,
        help="Minimum replay buffer size before " + "performing gradient updates.",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Minibatch size")
    parser.add_argument(
        "--render", action="store_true", help="Render env states in a GUI window."
    )
    parser.add_argument(
        "--demo", action="store_true", help="Just run evaluation, not training."
    )
    parser.add_argument("--load-pretrained", action="store_true", default=False)
    parser.add_argument(
        "--pretrained-type", type=str, default="best", choices=["best", "final"]
    )
    parser.add_argument(
        "--monitor", action="store_true", help="Wrap env with gym.wrappers.Monitor."
    )
    parser.add_argument(
        "--log-level", type=int, default=logging.INFO, help="Level of the root logger."
    )
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--ddpg-training-steps", type=int, default=int(1e3))
    parser.add_argument("--adversary-training-steps", type=int,default=int(1e3))
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)

    args.outdir = './results'
    print("Output files are saved in {}".format(args.outdir))

    # Set a random seed used in PFRL
    utils.set_random_seed(args.seed)

    def make_env(test):
        env = gym.make('DClawTurnFixed-v0')
        # Unwrap TimeLimit wrapper
        assert isinstance(env, gym.wrappers.TimeLimit)
        env = env.env
        # Use different random seeds for train and test envs
        env_seed = 2 ** 32 - 1 - args.seed if test else args.seed
        env.seed(env_seed)
        # Cast observations to float32 because our model uses float32
        env = pfrl.wrappers.CastObservationToFloat32(env)
        if args.monitor:
            env = pfrl.wrappers.Monitor(env, args.outdir)
        if args.render and not test:
            env = pfrl.wrappers.Render(env)
        return env

    env = make_env(test=False)
    timestep_limit = env.spec.max_episode_steps
    obs_space = env.observation_space
    action_space = env.action_space
    print("Observation space:", obs_space)
    print("Action space:", action_space)

    obs_size = obs_space.low.size
    action_size = action_space.low.size

    q_func = nn.Sequential(
        ConcatObsAndAction(),
        nn.Linear(obs_size + action_size, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256,256),
        nn.ReLU(),
        nn.Linear(256, 1),
    )
    policy = nn.Sequential(
        nn.Linear(obs_size, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256,256),
        nn.ReLU(),
        nn.Linear(256, action_size),
        BoundByTanh(low=action_space.low, high=action_space.high),
        DeterministicHead(),
    )

    ddpg_opt_a = torch.optim.Adam(policy.parameters())
    ddpg_opt_c = torch.optim.Adam(q_func.parameters())

    ddpg_rbuf = replay_buffers.ReplayBuffer(10 ** 6)

    ddpg_explorer = explorers.AdditiveGaussian(
        scale=0.1, low=action_space.low, high=action_space.high
    )

    def ddpg_burnin_action_func():
        """Select random actions until model is updated one or more times."""
        return np.random.uniform(action_space.low, action_space.high).astype(np.float32)

    # Hyperparameters in http://arxiv.org/abs/1802.09477
    ddpg_agent = DDPG(
        policy,
        q_func,
        ddpg_opt_a,
        ddpg_opt_c,
        ddpg_rbuf,
        gamma=args.gamma,
        explorer=ddpg_explorer,
        replay_start_size=args.replay_start_size,
        target_update_method="soft",
        target_update_interval=1,
        update_interval=1,
        soft_update_tau=5e-3,
        n_times_update=1,
        gpu=args.gpu,
        minibatch_size=args.batch_size,
        burnin_action_func=ddpg_burnin_action_func,
    )
    def adversary_random_func():
        return np.random.randint(0,9)
    # adversary_q = Critic(obs_size, 1, hidden_size=adversary_hidden_size)
    # adversary_action_space = gym.spaces.discrete.Discrete(9)
    # adversary_q = q_functions.FCQuadraticStateQFunction(
    #     obs_size, 1, n_hidden_channels = 256, n_hidden_layers = 2,action_space = adversary_action_space
    # )
    adversary_q = nn.Sequential(
        nn.Linear(obs_size, 256),
        nn.Linear(256,256),
        nn.Linear(256,256),
        nn.Linear(256,1),
        DiscreteActionValueHead(),
    )
    adversary_optimizer = torch.optim.Adam(adversary_q.parameters(), lr=1e-3)
    adversary_rbuf_capacity = int(1e6)
    adversary_rbuf = replay_buffers.ReplayBuffer(adversary_rbuf_capacity)
    adversary_explorer = explorers.LinearDecayEpsilonGreedy(
        1.0, 0.1, 10**4, adversary_random_func
    )

    adversary_agent = DQN(
        adversary_q,
        adversary_optimizer,
        adversary_rbuf,
        gpu=args.gpu,
        gamma=args.gamma,
        explorer=adversary_explorer,
        replay_start_size=args.replay_start_size,
        target_update_interval=1,
        minibatch_size=args.batch_size,
        target_update_method='soft',
        soft_update_tau=5e-3

    )
    logger = logging.getLogger(__name__)
    eval_env = make_env(test=True)
    evaluator = Evaluator(
        agent=ddpg_agent,
        n_steps=None,
        n_episodes=args.eval_n_runs,
        eval_interval=args.eval_interval,
        outdir=args.outdir,
        max_episode_len=timestep_limit,
        env=eval_env,
        step_offset=0,
        save_best_so_far_agent=True,
        use_tensorboard=True,
        logger=logger,
    )

    episode_reward = 0
    ddpg_episode_idx = 0
    adversary_episode_idx = 0

    # o_0, r_0
    current_state = env.reset()

    t = 0 
    ddpg_t = 0
    adversary_t = 0
    episode_len = 0
    try:
        while t < args.max_steps:
            for i in range(args.ddpg_training_steps):
                t += 1
                ddpg_t += 1
                ddpg_action = ddpg_agent.act(current_state)
                adversary_action = adversary_agent.act(current_state)
                ddpg_action[adversary_action] = 0
                next_state, reward, done, info = env.step(ddpg_action)
                episode_reward += reward
                episode_len += 1
                reset = episode_len == timestep_limit or info.get("needs_reset", False)
                ddpg_agent.observe(next_state, reward, done, reset)
                current_state = next_state
                if done or reset or t == args.max_steps:
                    logger.info(
                        "ddpg phase: outdir:%s step:%s episode:%s R:%s",
                        args.outdir,
                        ddpg_t,
                        ddpg_episode_idx,
                        episode_reward,
                    )
                    logger.info("statistics:%s", ddpg_agent.get_statistics())
                    if evaluator is not None:
                        evaluator.evaluate_if_necessary(t=t, episodes=ddpg_episode_idx + 1)
                    if t == args.max_steps:
                        break
                    episode_reward = 0
                    ddpg_episode_idx += 1
                    episode_len = 0
                    current_state = env.reset()
            episode_reward = 0
            episode_len = 0
            current_state = env.reset()
            print("start adversary training ")
            for i in range(args.adversary_training_steps):
                t += 1
                adversary_t += 1
                ddpg_action = ddpg_agent.act(current_state)
                adversary_action = adversary_agent.act(current_state)
                ddpg_action[adversary_action] = 0
                next_state, reward, done, info = env.step(ddpg_action)
                reward = -reward
                episode_len += 1
                reset = episode_len == timestep_limit or info.get("needs_reset", False)
                adversary_agent.observe(next_state, reward, done, reset)
                current_state = next_state

                if done or reset or t == args.max_steps:
                    if t == args.max_steps:
                        break
                    episode_reward = 0
                    adversary_episode_idx += 1
                    episode_len = 0
                    current_state = env.reset()
            

    except (Exception, KeyboardInterrupt):
        # Save the current model before being killed
        save_agent(ddpg_agent, t, args.outdir, logger, suffix="_ddpg_except")
        save_agent(adversary_agent, t, args.outdir, logger, suffix="_adversary_except" )
        raise

    # Save the final model
    save_agent(ddpg_agent, t, args.outdir, logger, suffix="_ddpg_finish")
    save_agent(adversary_agent, t, args.outdir, logger, suffix="_adversary_finish" )
    # if args.demo:
    #     eval_env.render()
    #     eval_stats = experiments.eval_performance(
    #         env=eval_env,
    #         agent=ddpg_agent,
    #         n_steps=None,
    #         n_episodes=args.eval_n_runbase_envs,
    #         max_episode_len=timestep_limit,
    #     )
    #     print(
    #         "n_runs: {} mean: {} median: {} stdev {}".format(
    #             args.eval_n_runs,
    #             eval_stats["mean"],
    #             eval_stats["median"],
    #             eval_stats["stdev"],
    #         )
    #     )
    # else:
    #     experiments.train_agent_with_evaluation(
    #         agent=ddpg_agent,
    #         env=env,
    #         steps=args.steps,
    #         eval_env=eval_env,
    #         eval_n_steps=None,
    #         eval_n_episodes=args.eval_n_runs,
    #         eval_interval=args.eval_interval,
    #         outdir=args.outdir,
    #         train_max_episode_len=timestep_limit,
    #     )
    print("finish")

        

    # episode_r = 0
    # episode_idx = 0

    # # o_0, r_0
    # obs = env.reset()

    # t = step_offset
    # if hasattr(agent, "t"):
    #     agent.t = step_offset

    # episode_len = 0
    # try:
    #     while t < max_steps:

    #         # a_t
    #         action = agent.act(obs)
    #         # o_{t+1}, r_{t+1}
    #         obs, r, done, info = env.step(action)
    #         t += 1
    #         episode_r += r
    #         episode_len += 1
    #         reset = episode_len == max_episode_len or info.get("needs_reset", False)
    #         agent.observe(obs, r, done, reset)


            # if done or reset or t == max_steps:
            #     logger.info(
            #         "outdir:%s step:%s episode:%s R:%s",
            #         outdir,
            #         t,
            #         episode_idx,
            #         episode_r,
            #     )
            #     logger.info("statistics:%s", agent.get_statistics())
            #     if evaluator is not None:
            #         evaluator.evaluate_if_necessary(t=t, episodes=episode_idx + 1)
            #         if (
            #             successful_score is not None
            #             and evaluator.max_score >= successful_score
            #         ):
            #             break
    #             if t == max_steps:
    #                 break
    #             # Start a new episode
    #             episode_r = 0
    #             episode_idx += 1
    #             episode_len = 0
    #             obs = env.reset()
    #         if checkpoint_freq and t % checkpoint_freq == 0:
    #             save_agent(agent, t, outdir, logger, suffix="_checkpoint")

    # except (Exception, KeyboardInterrupt):
    #     # Save the current model before being killed
    #     save_agent(agent, t, outdir, logger, suffix="_except")
    #     raise

    # # Save the final model
    # save_agent(agent, t, outdir, logger, suffix="_finish")

# def train_adversarial(
#     agent
#     env,
#     max_steps,
#     outdir,
#     checkpoint_freq=None,
#     max_episode_len=None,
#     step_offset=0,
#     evaluator=None,
#     successful_score=None,
#     logger=None,
# ):

#     logger = logger or logging.getLogger(__name__)

#     episode_r = 0
#     episode_idx = 0

#     # o_0, r_0
#     obs = env.reset()

#     t = step_offset
#     if hasattr(agent, "t"):
#         agent.t = step_offset

#     episode_len = 0
#     try:
#         while t < max_steps:

#             # a_t
#             action = agent.act(obs)
#             # o_{t+1}, r_{t+1}
#             obs, r, done, info = env.step(action)
#             t += 1
#             episode_r += r
#             episode_len += 1
#             reset = episode_len == max_episode_len or info.get("needs_reset", False)
#             agent.observe(obs, r, done, reset)


#             if done or reset or t == max_steps:
#                 logger.info(
#                     "outdir:%s step:%s episode:%s R:%s",
#                     outdir,
#                     t,
#                     episode_idx,
#                     episode_r,
#                 )
#                 logger.info("statistics:%s", agent.get_statistics())
#                 if evaluator is not None:
#                     evaluator.evaluate_if_necessary(t=t, episodes=episode_idx + 1)
#                     if (
#                         successful_score is not None
#                         and evaluator.max_score >= successful_score
#                     ):
#                         break
#                 if t == max_steps:
#                     break
#                 # Start a new episode
#                 episode_r = 0
#                 episode_idx += 1
#                 episode_len = 0
#                 obs = env.reset()
#             if checkpoint_freq and t % checkpoint_freq == 0:
#                 save_agent(agent, t, outdir, logger, suffix="_checkpoint")

#     except (Exception, KeyboardInterrupt):
#         # Save the current model before being killed
#         save_agent(agent, t, outdir, logger, suffix="_except")
#         raise

#     # Save the final model
#     save_agent(agent, t, outdir, logger, suffix="_finish")
if __name__ == "__main__":
   main()