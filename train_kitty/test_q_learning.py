import gym
import numpy as np
import torch
from modules import AdversarialDQN

import pdb
def eval_policy(policy, env_name, eval_episodes=10, seed = 0):
    env_seed = 2 ** 32 - 1 - seed
    eval_env = gym.make(env_name)
    eval_env.seed(env_seed)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(np.array(state), 'test')
            action = action[0]
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward
        
    avg_reward /= eval_episodes
    print("---------------------------------------")
    print("Evaluation over {} episodes: {:.3f}".format(eval_episodes, avg_reward))
    print("---------------------------------------")
    return avg_reward

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make("CartPole-v1")
    current_state = env.reset()
    state_dim = current_state.shape[0]
    n_actions = 2
    q_function = AdversarialDQN(state_dim=state_dim,
                                n_actions=n_actions,
                                device=device,
                                writer=None,
                                save_freq=100,
                                record_freq=100,
                                )
    start_learning = 100
    eval_freq = 100
    for t in range(20000):
        action = q_function.select_action(current_state, mode="train")
        action = action[0]
        next_state, reward, done, info = env.step(action)
        q_function.add_buffer(current_state, action, next_state, reward, done)
        if t == start_learning:
            print("start learning")
        if t > start_learning:
            q_function.train()
        current_state = next_state
        if done:
            current_state = env.reset()
        if t % eval_freq == 0:
            print('time steps: ',t)
            eval_policy(q_function, "CartPole-v1")
    