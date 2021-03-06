import torch
import numpy as np
import random

def trim_state(current_state):
    joint = current_state[:9]
    cossin = current_state[9:11]
    # command = current_state[11:20]
    other = current_state[20]
    output_state = np.concatenate((joint, cossin, np.array([other])), axis=0)
    return output_state
def kitty_trim_state(current_state):
    pose = current_state[:18]
    q_vel = current_state[24:36]
    output = np.concatenate((pose, q_vel))
    return output


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, outdir=None, max_size=int(1e6), device=torch.device('cuda')):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = torch.zeros((max_size, state_dim), device=device)
        self.action = torch.zeros((max_size, action_dim), device=device)
        self.next_state = torch.zeros((max_size, state_dim), device=device)
        self.reward = torch.zeros((max_size, 1), device=device)
        self.not_done = torch.zeros((max_size, 1), device=device)

        self.device = device
        self.outdir = outdir
    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = torch.tensor(state, device=self.device)
        self.action[self.ptr] = torch.tensor(action, device=self.device)
        self.next_state[self.ptr] = torch.tensor(next_state, device=self.device)
        self.reward[self.ptr] = torch.tensor(reward, device=self.device)
        self.not_done[self.ptr] = torch.tensor(1. - done, device=self.device)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            self.state[ind],
            self.action[ind],
            self.next_state[ind],
            self.reward[ind],
            self.not_done[ind]
        )
    def save(self, suffix=''):
        if self.size >= self.max_size:
            item = [self.state, self.action, self.next_state, self.reward, self.not_done, torch.tensor(self.ptr)]
            if self.outdir:
                torch.save(item, self.outdir + '/replay_buffer' + suffix)
            else:
                torch.save(item, './saved_models/replay_buffer' + suffix)
            print("finish saving the replay buffer")
        else:
            print("only save max buffer, the current size is {}/{}".format(self.size, self.max_size))
    def restore(self, suffix=''):
        if self.outdir:
            item = torch.load(self.outdir + '/replay_buffer' + suffix)
        else:
            item = torch.load('./saved_models/replay_buffer' + suffix)
        self.state = item[0]
        self.action = item[1]
        self.next_state = item[2]
        self.reward = item[3]
        self.not_done = item[4] 
        self.ptr = item[5]
        self.size = self.max_size


def generate_mask(episode_timestep, max_episode_timesteps):
    #episode_timestep [batchsize, timestep]
    batch_size = episode_timestep.shape[0]
    mask = torch.ones((batch_size, max_episode_timesteps))
    for i in range(batch_size):
        if episode_timestep[i] < max_episode_timesteps:
            mask[i,int(episode_timestep[i].item()):] = 0
    return mask  # shape:[batch, max time steps]

class ReplayMemory:
    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p

def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)
