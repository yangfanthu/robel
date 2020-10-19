import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import copy
import utils
import random
import pdb
from utils import soft_update, hard_update
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6
# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2

class MBCritic(torch.nn.Module):
    def __init__(self, state_dim_1:int, state_dim_2, action_dim:int, hidden_size: int = 256):
        super(MBCritic, self).__init__()
        self.state_dim_1 = state_dim_1
        self.state_dim_2 = state_dim_2
        self.action_dim	= action_dim
        self.hidden_size = hidden_size
        self.fc_1 = nn.Linear(state_dim_1 + state_dim_2 + action_dim, hidden_size)
        self.fc_2 = nn.Linear(hidden_size, hidden_size)
        self.fc_3 = nn.Linear(hidden_size, 1)

        self.fc_4 = nn.Linear(state_dim_1 + state_dim_2 + action_dim, hidden_size)
        self.fc_5 = nn.Linear(hidden_size, hidden_size)
        self.fc_6 = nn.Linear(hidden_size, 1)
    def forward(self, state, predict_state, action):
        xu = torch.cat((state, predict_state, action), dim=-1)

        x1 = F.relu(self.fc_1(xu))
        x1 = F.relu(self.fc_2(x1))
        x1 = self.fc_3(x1)

        x2 = F.relu(self.fc_4(xu))
        x2 = F.relu(self.fc_5(x2))
        x2 = self.fc_6(x2)
        return x1, x2


class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(DeterministicPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)


class MBSAC(object):
    def __init__(self, num_inputs, action_space, args, writer=None, outdir=None, device=torch.device("cpu")):
        self.index = 0
        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.writer = writer
        self.outdir = outdir
        self.batch_size = args.batch_size
        self.save_freq = args.save_freq

        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.device = device
        model = DynamicModel(input_state_dim=12, # remove broken info
                                  action_dim=action_dim,
                                  output_state_dim=9)
        model = model.to(device)
        self.model_wrapper = ModelWrapper(model = model, broken_angle=self.broken_angle)
        self.mse_loss = nn.MSELoss()
        self.model_optimizer = torch.optim.Adam(self.model_wrapper.model.parameters(), lr = args.lr)

        self.replay_buffer = utils.ReplayMemory(capacity=args.buffer_max_size, seed=args.seed)
        self.model_replay_buffer = utils.ReplayBuffer(state_dim=state_dim, action_dim=action_dim, max_size=int(buffer_max_size / 10), device=device, outdir=outdir)

        self.critic = MBQNetwork(num_inputs, 9, action_space.shape[0], args.hidden_size).to(device=self.device)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=args.lr)

        self.critic_target = MBQNetwork(num_inputs, 9, action_space.shape[0], args.hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=args.lr)

            self.policy = GaussianPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=args.lr)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=args.lr)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]
    
    def add_buffer(self, state, action, next_state, reward,  done):
        self.replay_buffer.push(state, action, next_state, reward, done)
    def add_model_buffer(self, current_state, action, next_state, reward, done):
        self.model_replay_buffer.add(current_state, action, next_state, reward, done)
    def update_parameters(self):
        # Sample a batch from memory
        state_batch, action_batch, next_state_batch, reward_batch, mask_batch = self.replay_buffer.sample(batch_size=self.batch_size)
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)


        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()


        current_state, action, next_state, _, __ = self.model_replay_buffer.sample(self.batch_size)
        predict_next_state = self.model_wrapper.forward(current_state, action)
        # pdb.set_trace()
        # predict_next_state = predict_next_state[:,:-9]
        next_state = next_state[:,:9]
        model_loss = self.mse_loss(predict_next_state, next_state)
        self.model_optimizer.zero_grad()
        model_loss.backward()
        self.model_optimizer.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        if self.index % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        if self.writer:
            self.writer.add_scalar('loss/critic_1', qf1_loss.item(), self.index)
            self.writer.add_scalar('loss/critic_2', qf2_loss.item(), self.index)
            self.writer.add_scalar('loss/policy', policy_loss.item(), self.index)
            self.writer.add_scalar('loss/entropy_loss', alpha_loss.item(), self.index)
            self.writer.add_scalar('entropy_temprature/alpha', alpha_tlogs.item(), self.index)
            self.writer.add_scalar('./train/model_loss', model_loss.mean().item(), self.index)

        if self.index % self.save_freq == 0:
            self.save_model()

        self.index += 1
        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    def save_model(self):
        print('saving...')
        if self.outdir != None:
            torch.save(self.policy.state_dict(),self.outdir + '/actor_{:07d}.ckpt'.format(self.index))
            torch.save(self.critic.state_dict(),self.outdir + '/critic_{:07d}.ckpt'.format(self.index))
            torch.save(self.critic_target.state_dict(),self.outdir + '/critic_target_{:07d}.ckpt'.format(self.index))
        else:
            torch.save(self.policy.state_dict(),'./saved_models/actor_{:07d}.ckpt'.format(self.index))
            torch.save(self.critic.state_dict(),'./saved_models/critic_{:07d}.ckpt'.format(self.index))
            torch.save(self.critic_target.state_dict(),'./saved_models/critic_target_{:07d}.ckpt'.format(self.index))
        # self.replay_buffer.save()
        print('finish saving')
    # Load model parameters
    def load_model(self, actor_path, critic_path):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.policy.load_state_dict(torch.load(actor_path, map_location=self.device))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path, map_location=self.device))
    def restore_model_for_test(self, index):
        self.policy.load_state_dict(torch.load('./saved_models/actor_{:07d}.ckpt'.format(index), map_location = self.device))
        # self.critic.load_state_dict(torch.load('./saved_models/critic_{:07d}.ckpt'.format(index), map_location = self.device))
        print('finish restoring model')
    def load_pretrained_model(self):
        self.model_wrapper.model.load_state_dict(torch.load('./saved_models/model_033.ckpt', map_location=self.device))
        print("finish loading the model")
class SAC(object):
    def __init__(self, num_inputs, action_space, args, writer=None, outdir=None, device=torch.device("cpu")):
        self.index = 0
        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.writer = writer
        self.outdir = outdir
        self.batch_size = args.batch_size
        self.save_freq = args.save_freq

        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.device = device

        self.replay_buffer = utils.ReplayMemory(capacity=args.buffer_max_size, seed=args.seed)

        self.critic = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(device=self.device)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=args.lr)

        self.critic_target = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=args.lr)

            self.policy = GaussianPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=args.lr)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=args.lr)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]
    
    def add_buffer(self, state, action, next_state, reward,  done):
        self.replay_buffer.push(state, action, next_state, reward, done)

    def update_parameters(self):
        # Sample a batch from memory
        state_batch, action_batch, next_state_batch, reward_batch, mask_batch = self.replay_buffer.sample(batch_size=self.batch_size)
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)


        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        if self.index % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        if self.writer:
            self.writer.add_scalar('loss/critic_1', qf1_loss.item(), self.index)
            self.writer.add_scalar('loss/critic_2', qf2_loss.item(), self.index)
            self.writer.add_scalar('loss/policy', policy_loss.item(), self.index)
            self.writer.add_scalar('loss/entropy_loss', alpha_loss.item(), self.index)
            self.writer.add_scalar('entropy_temprature/alpha', alpha_tlogs.item(), self.index)
        if self.index % self.save_freq == 0:
            self.save_model()

        self.index += 1
        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    def save_model(self):
        print('saving...')
        if self.outdir != None:
            torch.save(self.policy.state_dict(),self.outdir + '/actor_{:07d}.ckpt'.format(self.index))
            torch.save(self.critic.state_dict(),self.outdir + '/critic_{:07d}.ckpt'.format(self.index))
            torch.save(self.critic_target.state_dict(),self.outdir + '/critic_target_{:07d}.ckpt'.format(self.index))
        else:
            torch.save(self.policy.state_dict(),'./saved_models/actor_{:07d}.ckpt'.format(self.index))
            torch.save(self.critic.state_dict(),'./saved_models/critic_{:07d}.ckpt'.format(self.index))
            torch.save(self.critic_target.state_dict(),'./saved_models/critic_target_{:07d}.ckpt'.format(self.index))
        # self.replay_buffer.save()
        print('finish saving')
    # Load model parameters
    def load_model(self, actor_path, critic_path):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.policy.load_state_dict(torch.load(actor_path, map_location=self.device))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path, map_location=self.device))
    def restore_model_for_test(self, index):
        self.policy.load_state_dict(torch.load('./saved_models/actor_{:07d}.ckpt'.format(index), map_location = self.device))
        # self.critic.load_state_dict(torch.load('./saved_models/critic_{:07d}.ckpt'.format(index), map_location = self.device))
        print('finish restoring model')


class Critic(torch.nn.Module):
    def __init__(self, state_dim:int , action_dim:int, hidden_size: int = 256):
        super(Critic, self).__init__()
        self.state_dim = state_dim
        self.action_dim	= action_dim
        self.hidden_size = hidden_size
        self.fc_1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.fc_2 = nn.Linear(hidden_size, hidden_size)
        self.fc_3 = nn.Linear(hidden_size, hidden_size)
        self.fc_4 = nn.Linear(hidden_size, 1)
    def forward(self, state, action):
        hidden = torch.cat((state,action), dim=-1)
        hidden = F.relu(self.fc_1(hidden))
        hidden = F.relu(self.fc_2(hidden))
        hidden = F.relu(self.fc_3(hidden))
        q_value = self.fc_4(hidden)
        return q_value
class MBCritic(torch.nn.Module):
    def __init__(self, state_dim_1:int, state_dim_2, action_dim:int, hidden_size: int = 256):
        super(MBCritic, self).__init__()
        self.state_dim_1 = state_dim_1
        self.state_dim_2 = state_dim_2
        self.action_dim	= action_dim
        self.hidden_size = hidden_size
        self.fc_1 = nn.Linear(state_dim_1 + state_dim_2 + action_dim, hidden_size)
        self.fc_2 = nn.Linear(hidden_size, hidden_size)
        self.fc_3 = nn.Linear(hidden_size, hidden_size)
        self.fc_4 = nn.Linear(hidden_size, 1)
    def forward(self, state, predict_state, action):
        hidden = torch.cat((state, predict_state, action), dim=-1)
        hidden = F.relu(self.fc_1(hidden))
        hidden = F.relu(self.fc_2(hidden))
        hidden = F.relu(self.fc_3(hidden))
        q_value = self.fc_4(hidden)
        return q_value

class Actor(torch.nn.Module):
    def __init__(self, state_dim: int, action_dim:int, max_action:float = 1.,hidden_size: int = 256, broken_info_recap = False):
        super(Actor, self).__init__()
        self.max_action = max_action
        self.state_dim = state_dim
        self.action_dim	= action_dim
        self.hidden_size = hidden_size
        self.broken_info_recap = broken_info_recap

        self.fc_1 = nn.Linear(state_dim, hidden_size)
        self.fc_2 = nn.Linear(hidden_size, hidden_size)
        self.fc_3 = nn.Linear(hidden_size, hidden_size)
        self.fc_4 = nn.Linear(hidden_size, action_dim)
    def forward(self, state):
        if self.broken_info_recap:
            if len(state.shape) == 1:
                joint_info = state[-9:]
            elif len(state.shape) == 2:
                joint_info = state[:, -9:]
            else:
                raise ValueError("input the wrong state for the actor")
        hidden = F.relu(self.fc_1(state))
        hidden = F.relu(self.fc_2(hidden))
        hidden = F.relu(self.fc_3(hidden))
        hidden = self.fc_4(hidden)
        action = torch.tanh(hidden) * self.max_action
        if True in torch.isnan(action):
            pdb.set_trace()
        if self.broken_info_recap:
            action = action * joint_info
        return action

class DDPG(object):
    def __init__(self, 
        state_dim:int,
        action_dim:int, 
        device, 
        writer,
        buffer_max_size=int(1e6), 
        gamma:float = 0.99, 
        batch_size:int = 64, 
        max_action:float = 1., 
        hidden_size: int = 256,
        tau = 0.005,
        variance:float = 0.1,
        save_freq:int = 8000,
        record_freq:int=100,
        outdir = None,
        broken_info_recap = False):

        self.batch_size = batch_size
        self.gamma = gamma
        self.max_action = max_action
        self.hidden_size = hidden_size
        self.replay_buffer = utils.ReplayBuffer(state_dim = state_dim, action_dim = action_dim, max_size=buffer_max_size, device=device, outdir=outdir)
        self.tau = tau
        self.device = device
        self.variance = variance
        self.action_dim = action_dim
        self.state_dim	= state_dim
        self.save_freq = save_freq
        self.record_freq = record_freq
        self.broken_info_recap = broken_info_recap
        self.critic = Critic(state_dim = state_dim, action_dim = action_dim, hidden_size = hidden_size).to(device)
        self.critic_target = copy.deepcopy(self.critic).to(device)
        self.critic_target.eval()
        self.actor = Actor(state_dim = state_dim, action_dim = action_dim, hidden_size = hidden_size, max_action = max_action, broken_info_recap=broken_info_recap).to(device)
        self.actor_target = copy.deepcopy(self.actor).to(device)
        self.actor_target.eval()

        self.mse_loss = nn.MSELoss()
        self.index = 0
        self.writer = writer
        learning_rate = 1e-3
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr = 1e-3)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr = 1e-3)
        self.outdir = outdir

    def add_buffer(self,current_state,action,next_state,reward,done):
        self.replay_buffer.add(current_state,action,next_state,reward,done)
    def select_action(self, state, mode = "train"):
        temp_state = state
        state = torch.tensor(state).float().to(self.device)
        with torch.no_grad():
            if mode == "train":
                action = self.actor(state).cpu().detach().numpy()
                noise = np.random.normal(0,self.variance,self.action_dim)
                action = action + noise
            elif mode == "test":
                action = self.actor(state).cpu().detach().numpy()
        action = np.clip(action, -self.max_action, self.max_action)
        return action
    def save_model(self):
        print('saving...')
        if self.outdir != None:
            torch.save(self.actor.state_dict(),self.outdir + '/actor_{:07d}.ckpt'.format(self.index))
            torch.save(self.critic.state_dict(),self.outdir + '/critic_{:07d}.ckpt'.format(self.index))
            torch.save(self.actor_target.state_dict(),self.outdir + '/actor_target_{:07d}.ckpt'.format(self.index))
            torch.save(self.critic_target.state_dict(),self.outdir + '/critic_target_{:07d}.ckpt'.format(self.index))
        else:
            torch.save(self.actor.state_dict(),'./saved_models/actor_{:07d}.ckpt'.format(self.index))
            torch.save(self.critic.state_dict(),'./saved_models/critic_{:07d}.ckpt'.format(self.index))
            torch.save(self.actor_target.state_dict(),'./saved_models/actor_target_{:07d}.ckpt'.format(self.index))
            torch.save(self.critic_target.state_dict(),'./saved_models/critic_target_{:07d}.ckpt'.format(self.index))
        self.replay_buffer.save()
        print('finish saving')
    def restore_model_for_test(self, index):
        self.actor.load_state_dict(torch.load('./saved_models/actor_{:07d}.ckpt'.format(index), map_location = self.device))
        self.critic.load_state_dict(torch.load('./saved_models/critic_{:07d}.ckpt'.format(index), map_location = self.device))
        print('finish restoring model')
    def restore_model_for_train(self, index):
        self.index = index
        self.actor.load_state_dict(torch.load('./saved_models/actor_{:07d}.ckpt'.format(index), map_location = self.device))
        self.actor_target.load_state_dict(torch.load('./saved_models/actor_target_{:07d}.ckpt'.format(index), map_location = self.device))
        self.critic.load_state_dict(torch.load('./saved_models/critic_{:07d}.ckpt'.format(index), map_location = self.device))
        self.critic_target.load_state_dict(torch.load('./saved_models/critic_target_{:07d}.ckpt'.format(index), map_location = self.device))		
        print('finish restoring model')

    def train(self):
        self.index += 1
        current_state, action, next_state, reward, not_done = self.replay_buffer.sample(self.batch_size)
        
        self.critic_optimizer.zero_grad()
        with torch.no_grad():
            target = reward + self.gamma * not_done * self.critic_target(next_state, self.actor_target(next_state))
        current_q = self.critic(current_state, action)
        critic_loss = self.mse_loss(target, current_q)
        

        critic_loss.backward()
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        actor_loss = - self.critic(current_state, self.actor(current_state)).mean()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.index % self.record_freq == 0 and self.writer != None:
            self.writer.add_scalar('./train/critic_loss',critic_loss.item(), self.index)
            self.writer.add_scalar('./train/actor_loss', actor_loss.item(), self.index)
            self.writer.add_scalar('./train/current_q', current_q.mean().item(), self.index)
            self.writer.add_scalar('./train/reward_max', reward.max().item(), self.index)
            self.writer.add_scalar('./train/reward_mean', reward.mean().item(), self.index)
            self.writer.add_scalar('./train/actor_q', -actor_loss.item(), self.index)


        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        if self.index % self.save_freq == 0:
            self.save_model()

class AdversarialDQN(object):
    def __init__(self, 
        state_dim, n_actions, 
        device, 
        writer, 
        buffer_max_size=int(1e6),
        epsilon=0.1, 
        hidden_size=256, 
        batch_size=64, 
        learning_rate=1e-3,
        gamma=0.9,
        tau=5e-3,
        save_freq=2000,
        record_freq=1000,
        outdir=None):

        self.state_dim = state_dim # state space is the same as the ddpg agent
        self.n_actions = n_actions
        self.hidden_size = hidden_size
        self.q_function = Critic(state_dim, 1, hidden_size=hidden_size).to(device)
        self.q_target = copy.deepcopy(self.q_function).to(device)
        self.epsilon = epsilon
        self.device = device
        self.writer = writer
        self.replay_buffer = utils.ReplayBuffer(state_dim=state_dim, action_dim=1, outdir=outdir, max_size=buffer_max_size, device=device)
        self.index = 0
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.q_function.parameters(), lr = learning_rate)
        self.gamma = gamma
        self.tau = tau
        self.record_freq = record_freq
        self.save_freq = save_freq
        self.criterion = torch.nn.MSELoss()
        self.outdir = outdir
    def add_buffer(self,current_state,action,next_state,reward,done):
        self.replay_buffer.add(current_state,action,next_state,reward,done)
    def select_action(self, state, mode = "train"):
        state = torch.tensor(state).float().to(self.device)
        scores = [self.q_function(state, torch.tensor([action]).float().to(self.device)).cpu().item() for action in range(self.n_actions)]
        scores = np.array(scores)
        action = np.where(scores == scores.max())[0][0]
        action = np.array([action])
        if mode == "train":
            dice = random.random()
            if dice < self.epsilon:
                action = np.random.randint(low = 0,high = self.n_actions, size = 1)
        # if dice < self.epsilon:
        # 	action = np.random.randint(low = 0,high = self.n_actions, size = 1)
        # else:
        # 	scores = [self.q_function(state, torch.tensor([action]).float().to(self.device)).cpu().item() for action in range(self.n_actions)]
        # 	scores = np.array(scores)
        # 	action = np.where(scores == scores.max())[0][0]
        # 	action = np.array([action])
        return action
    def select_action_target(self, state):
        # scores = []
        # for current_action in range(self.n_actions):
        # 	batch_action = np.ones((self.batch_size, 1)) * current_action
        # 	batch_action = torch.tensor(batch_action).float()
        # 	current_score = self.q_function(state, batch_action).cpu()
        # 	scores.append(current_score)
        # pdb.set_trace()
        scores = [np.array(self.q_target(state, torch.tensor(np.ones((self.batch_size, 1))*action).float().to(self.device)).cpu())for action in range(self.n_actions)]
        scores = np.array(scores)
        scores = scores.transpose(1,0,2)
        max_scores = scores.max(1)
        action_list = []
        for i in range(self.batch_size):
            action_loc = np.where(scores[i]==max_scores[i])[0][0]
            action_list.append(action_loc)
        action = np.array(action_list)
        action = torch.tensor(action).float().unsqueeze(1).to(self.device)
        return action
    def save_model(self):
        print("saving the adversarial q function....")
        if self.outdir != None:
            torch.save(self.q_function.state_dict(), self.outdir + '/advesaral_q_{:07d}.ckpt'.format(self.index))
            torch.save(self.q_target.state_dict(), self.outdir + '/advesaral_q_target_{:07d}.ckpt'.format(self.index))
        else:
            torch.save(self.q_function.state_dict(),'./saved_models/advesaral_q_{:07d}.ckpt'.format(self.index))
            torch.save(self.q_target.state_dict(),'./saved_models/advesaral_q_target_{:07d}.ckpt'.format(self.index))
        self.replay_buffer.save('adversary')
        print("finish saving the advesarial model")
    def restore_model(self, index):
        self.index = index
        self.q_function.load_state_dict(torch.load('./saved_models/advesaral_q_{:07d}.ckpt'.format(self.index), map_location = self.device))
        self.q_target.load_state_dict(torch.load('./saved_models/advesaral_q_target_{:07d}.ckpt'.format(self.index), map_location = self.device))
        print("finish restoring the advesarial model")
    def train(self):
        self.index += 1
        current_state, action, next_state, reward, not_done = self.replay_buffer.sample(self.batch_size)

        self.optimizer.zero_grad()
        
        with torch.no_grad():
            target = reward + self.gamma * not_done * self.q_target(next_state, self.select_action_target(next_state))
        current_q = self.q_function(current_state, action)
        loss = self.criterion(current_q, target)

        loss.backward()
        self.optimizer.step()
        
        if self.index % self.record_freq == 0 and self.writer != None:
            self.writer.add_scalar('./train/adversarial_loss', loss.cpu().item(), self.index)
            self.writer.add_scalar('./train/advesarial_q', current_q.cpu().mean(), self.index)

        for param, target_param in zip(self.q_function.parameters(), self.q_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        if self.index % self.save_freq == 0:
            self.save_model()

class DynamicModel(torch.nn.Module):
    def __init__(self, input_state_dim, action_dim, output_state_dim, hidden_dim=512):
        super(DynamicModel, self).__init__()
        self.fc_1 = nn.Linear(input_state_dim + action_dim, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_4 = nn.Linear(hidden_dim, output_state_dim)
    def forward(self, state, action):
        x = torch.cat((state, action), dim=-1)
        hidden = F.relu(self.fc_1(x))
        hidden = F.relu(self.fc_2(hidden))
        hidden = F.relu(self.fc_3(hidden))
        output = self.fc_4(hidden)
        return output
class ModelWrapper(object):
    def __init__(self, model, broken_angle = -0.6):
        self.model = model
        self.broken_angle = broken_angle
    def forward(self, state, action):
        if len(state.shape) == 1:
            joint_info = state[-9:]
            input_state = state[:-9]
        elif len(state.shape) == 2:
            joint_info = state[:, -9:]
            input_state = state[:, :-9]
        broken_joints = torch.where(joint_info == 0)
        action[broken_joints] = self.broken_angle
        output = self.model.forward(input_state, action)
        return output

class MBDDPG(object):
    def __init__(self, 
        state_dim:int,
        state_dim_2:int,
        action_dim:int, 
        device, 
        writer,
        buffer_max_size=int(1e6), 
        gamma:float = 0.99, 
        batch_size:int = 64, 
        max_action:float = 1., 
        hidden_size: int = 256,
        tau = 0.005,
        variance:float = 0.1,
        save_freq:int = 8000,
        record_freq:int=100,
        outdir = None,
        broken_info_recap = False,
        broken_angle = -0.6):

        self.batch_size = batch_size
        self.gamma = gamma
        self.max_action = max_action
        self.hidden_size = hidden_size
        self.replay_buffer = utils.ReplayBuffer(state_dim=state_dim, action_dim=action_dim, max_size=buffer_max_size, device=device, outdir=outdir)
        self.model_replay_buffer = utils.ReplayBuffer(state_dim=state_dim, action_dim=action_dim, max_size=int(buffer_max_size / 10), device=device, outdir=outdir)
        self.tau = tau
        self.device = device
        self.variance = variance
        self.action_dim = action_dim
        self.state_dim	= state_dim
        self.save_freq = save_freq
        self.record_freq = record_freq
        self.broken_info_recap = broken_info_recap
        # self.critic = Critic(state_dim = state_dim, action_dim = 9, hidden_size = hidden_size).to(device)
        self.critic = MBCritic(state_dim_1 = state_dim, state_dim_2=state_dim_2, action_dim = action_dim, hidden_size = hidden_size).to(device)
        self.critic_target = copy.deepcopy(self.critic).to(device)
        self.critic_target.eval()
        self.actor = Actor(state_dim = state_dim, action_dim = action_dim, hidden_size = hidden_size, max_action = max_action, broken_info_recap=broken_info_recap).to(device)
        self.actor_target = copy.deepcopy(self.actor).to(device)
        self.actor_target.eval()
        self.broken_angle = broken_angle
        output_state_dim = state_dim
        model = DynamicModel(input_state_dim=13, # remove broken info
                                  action_dim=9,
                                  output_state_dim=9)
        model = model.to(device)
        self.model_wrapper = ModelWrapper(model = model, broken_angle=self.broken_angle)
        self.mse_loss = nn.MSELoss()
        self.index = 0
        self.writer = writer
        learning_rate = 1e-3
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr = 1e-3)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr = 1e-3)
        self.model_optimizer = torch.optim.Adam(self.model_wrapper.model.parameters(), lr = 1e-3)
        self.outdir = outdir
    def load_pretrained_model(self):
        self.model_wrapper.model.load_state_dict(torch.load('./saved_models/model_033.ckpt', map_location=self.device))
        print("finish loading the model")

    def add_buffer(self,current_state,action,next_state,reward,done):
        self.replay_buffer.add(current_state, action, next_state, reward, done)
    def add_model_buffer(self, current_state, action, next_state, reward, done):
        self.model_replay_buffer.add(current_state, action, next_state, reward, done)
    def select_action(self, state, mode = "train"):
        temp_state = state
        state = torch.tensor(state).float().to(self.device)
        with torch.no_grad():
            if mode == "train":
                action = self.actor(state).cpu().detach().numpy()
                noise = np.random.normal(0, self.variance, self.action_dim)
                action = action + noise
            elif mode == "test":
                action = self.actor(state).cpu().detach().numpy()
        action = np.clip(action, -self.max_action, self.max_action)
        return action
    def save_model(self):
        print('saving...')
        if self.outdir != None:
            torch.save(self.actor.state_dict(),self.outdir + '/actor_{:07d}.ckpt'.format(self.index))
            torch.save(self.critic.state_dict(),self.outdir + '/critic_{:07d}.ckpt'.format(self.index))
            torch.save(self.actor_target.state_dict(),self.outdir + '/actor_target_{:07d}.ckpt'.format(self.index))
            torch.save(self.critic_target.state_dict(),self.outdir + '/critic_target_{:07d}.ckpt'.format(self.index))
            torch.save(self.model_wrapper.model.state_dict(), self.outdir + '/model{:07d}.ckpt'.format(self.index))
        else:
            torch.save(self.actor.state_dict(),'./saved_models/actor_{:07d}.ckpt'.format(self.index))
            torch.save(self.critic.state_dict(),'./saved_models/critic_{:07d}.ckpt'.format(self.index))
            torch.save(self.actor_target.state_dict(),'./saved_models/actor_target_{:07d}.ckpt'.format(self.index))
            torch.save(self.critic_target.state_dict(),'./saved_models/critic_target_{:07d}.ckpt'.format(self.index))
            torch.save(self.model_wrapper.model.state_dict(), './saved_models/model_{:07d}.ckpt'.format(self.index))
        self.replay_buffer.save()
        self.model_replay_buffer.save('model')
        print('finish saving')
    def restore_model_for_test(self, index):
        self.actor.load_state_dict(torch.load('./saved_models/actor_{:07d}.ckpt'.format(index), map_location = self.device))
        self.critic.load_state_dict(torch.load('./saved_models/critic_{:07d}.ckpt'.format(index), map_location = self.device))
        print('finish restoring model')
    def restore_model_for_train(self, index):
        self.index = index
        self.actor.load_state_dict(torch.load('./saved_models/actor_{:07d}.ckpt'.format(index), map_location = self.device))
        self.actor_target.load_state_dict(torch.load('./saved_models/actor_target_{:07d}.ckpt'.format(index), map_location = self.device))
        self.critic.load_state_dict(torch.load('./saved_models/critic_{:07d}.ckpt'.format(index), map_location = self.device))
        self.critic_target.load_state_dict(torch.load('./saved_models/critic_target_{:07d}.ckpt'.format(index), map_location = self.device))		
        print('finish restoring model')

    def train(self):
        self.index += 1
        current_state, action, next_state, reward, not_done = self.replay_buffer.sample(self.batch_size)
        
        self.critic_optimizer.zero_grad()
        with torch.no_grad():
            target = reward + self.gamma * not_done * \
                self.critic_target(next_state, self.model_wrapper.forward(next_state, self.actor_target(next_state)), self.actor_target(next_state))
        current_q = self.critic(current_state, self.model_wrapper.forward(current_state, action), action)
        critic_loss = self.mse_loss(target, current_q)
        

        critic_loss.backward()
        self.critic_optimizer.step()

        # self.actor_optimizer.zero_grad()
        # actor_loss = -self.critic(current_state, self.model_wrapper.forward(current_state, self.actor(current_state)), self.actor(current_state)).mean()
        # actor_loss.backward()
        # self.actor_optimizer.step()
        current_state[(self.state_dim-9):] = 1
        broken_index = [0,1,2,3,4,5,6,7,8]
        random.shuffle(broken_index)
        for i in broken_index:
            current_state[self.state_dim - 9 + i] = 0
            self.actor_optimizer.zero_grad()
            actor_loss = -self.critic(current_state, self.model_wrapper.forward(current_state, self.actor(current_state)), self.actor(current_state)).mean()
            actor_loss.backward()
            self.actor_optimizer.step()
        # TODO: whether we should loop over every broken case to train actor

        current_state, action, next_state, reward, not_done = self.model_replay_buffer.sample(self.batch_size)
        predict_next_state = self.model_wrapper.forward(current_state, action)
        # pdb.set_trace()
        # predict_next_state = predict_next_state[:,:-9]
        next_state = next_state[:,:9]
        model_loss = self.mse_loss(predict_next_state, next_state)
        self.model_optimizer.zero_grad()
        model_loss.backward()
        self.model_optimizer.step()

        if self.index % self.record_freq == 0 and self.writer != None:
            self.writer.add_scalar('./train/critic_loss',critic_loss.item(), self.index)
            self.writer.add_scalar('./train/actor_loss', actor_loss.item(), self.index)
            self.writer.add_scalar('./train/current_q', current_q.mean().item(), self.index)
            self.writer.add_scalar('./train/reward_max', reward.max().item(), self.index)
            self.writer.add_scalar('./train/reward_mean', reward.mean().item(), self.index)
            self.writer.add_scalar('./train/model_loss', model_loss.mean().item(), self.index)
            # self.writer.add_scalar('./train/actor_q', -actor_loss.item(), self.index)


        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        if self.index % self.save_freq == 0:
            self.save_model()


if __name__ == "__main__":
    adversarial_dqn = AdversarialDQN(state_dim=3, n_actions=3,device='cpu',writer=None)
    dumb_state = np.ones(3)
    dumb_next_state = np.ones(3)
    dumb_action = np.array([1])
    dumb_reward = 10
    adversarial_dqn.add_buffer(dumb_state, dumb_action, dumb_next_state, dumb_reward, False)
    adversarial_dqn.train()
    print('finish')


        




