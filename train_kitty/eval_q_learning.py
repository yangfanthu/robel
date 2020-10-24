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

    q_function.restore_model(19200)
    # q_function.q_function.load_state_dict(torch.load('./saved_models/advesaral_q_0009800.ckpt'))
    start_learning = 100
    eval_freq = 100
    sum_reward = 0
    
    for t in range(10000):
        env.render()
        action = q_function.select_action(current_state, mode="test")
        action = action[0]
        # action = 1
        next_state, reward, done, info = env.step(action)
        sum_reward += reward
        current_state = next_state
        if done:
            print(sum_reward)
            sum_reward = 0
            current_state = env.reset()
    # for _ in range(1000):
    #     env.render()
    #     env.step(env.action_space.sample()) # take a random action
    # env.close()
    