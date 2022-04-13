'''
tools used to compute the posterior over the parameters of a state-space model given some trajectory
'''

import torch
from torch import nn
from torchdiffeq import odeint
from utils import MLP, NormalMixture1D

def dy_dt(t, y, theta):
    dy_dt = torch.matmul(y, theta)
    return dy_dt

def solve_ode_sample(batch_size, n_vars, y_init_range, theta_range, noise_std_ratio, ts):
    y_init = torch.distributions.Uniform(*y_init_range).rsample((batch_size, n_vars))
    theta_dist = torch.distributions.Uniform(*theta_range)
    theta = theta_dist.rsample((n_vars, n_vars))

    y_mean = odeint(lambda t, y: dy_dt(t,y, theta=theta), y_init, ts, rtol=1e-6, atol=1e-5)
    y_obs = torch.distributions.Normal(loc=y_mean, scale=noise_std_ratio*torch.abs(y_mean)).rsample()
    y_obs = y_obs.swapaxes(0,1).reshape((batch_size, n_vars*len(ts)))

    return y_obs

class BatchedNormalMixture1D():
    def __init__(self, mixture_probs, means, stds):
        self.num_mixtures = mixture_probs.shape[-1]
        self.normals = torch.distributions.Normal(loc=means, scale=stds)
        self.mixture_probs = mixture_probs

    def log_prob(self, batched_obs):
        batched_obs = batched_obs.reshape(-1, 1).expand(-1,self.num_mixtures)
        normal_log_prob = self.normals.log_prob(batched_obs)
        category_log_prob = torch.log(self.mixture_probs)
        return torch.logsumexp(normal_log_prob + category_log_prob, dim=1)


class InferenceNetworkPQ_NormalMixture(nn.Module):
    def __init__(self, num_mixtures, input_size, num_layers, num_hidden_units):
        super().__init__()
        self.num_mixtures = num_mixtures
        self.mixture_probs_net = MLP(num_layers, num_hidden_units, input_size,
                                     output_size=num_mixtures,
                                     activation= nn.Tanh(),
                                     last_activation=torch.exp)
        self.mu_net, self.offset_net = [MLP(num_layers, num_hidden_units, input_size)]*2
        self.sigma_net = MLP(num_layers, num_hidden_units, input_size, last_activation=torch.exp)

    def get_q_x_given_obs(self, obs):
        mu = self.mu_net(obs)
        sigma = self.sigma_net(obs)
        offset = self.offset_net(obs)
        mixture_probs = self.mixture_probs_net(obs)

        means = torch.stack([mu + offset*i for i in range(self.num_mixtures)], dim=1)
        stds = torch.stack([sigma]*self.num_mixtures, dim=1)

        return BatchedNormalMixture1D(mixture_probs/mixture_probs.sum(dim=1).unsqueeze(-1), means, stds)

    def forward(self, theta, obs):
        q_x_given_obs = self.get_q_x_given_obs(obs)
        log_prob = q_x_given_obs.log_prob(theta)
        return -torch.mean(log_prob)



