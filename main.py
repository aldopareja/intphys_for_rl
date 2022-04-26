'''
tools used to compute the posterior over the parameters of a state-space model given some trajectory
'''

import torch
from torch import nn
from torchdiffeq import odeint
from utils import MLP, NormalMixture1D

def dy_dt(y, theta):
    dy_dt = torch.matmul(y, theta)
    return dy_dt

def solve_ode_sample(batch_size, n_vars, y_init_range, theta_range, noise_std_ratio, ts, device='cpu'):
    y_init = torch.distributions.Uniform(*torch.tensor(y_init_range,device=device)).rsample((batch_size, n_vars))
    theta_dist = torch.distributions.Uniform(*torch.tensor(theta_range,device=device))
    theta = theta_dist.rsample((n_vars, n_vars))
    ts = ts.to(device)

    y_mean = odeint(lambda t, y: dy_dt(y, theta=theta), y_init, ts, rtol=1e-6, atol=1e-5)
    y_obs = torch.distributions.Normal(loc=y_mean, scale=noise_std_ratio*torch.abs(y_mean)).rsample()
    y_obs = y_obs.swapaxes(0,1)

    return theta, y_obs

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

class SeqGaussMixPosterior(nn.Module):
    def __init__(self, num_mixtures):
        super().__init__()
        self.embed_obs = MLP(2, 10, output_size=10, unsqueeze=False)
        self.decode_mu = MLP(1, 12, input_size=10,
                             last_activation=lambda y: torch.tanh(y) * 10,
                             unsqueeze=False)
        self.decode_sigma = MLP(1, 12, input_size=10,
                                last_activation=torch.exp,
                                unsqueeze=False)
        self.decode_mixture_prob = MLP(1, 12, input_size=10,
                                       last_activation=torch.exp,
                                       unsqueeze=False)
        self.trans_dec = nn.TransformerDecoderLayer(d_model=10, nhead=2, dim_feedforward=20,
                                                    batch_first=True, dropout=0.0)

        self.start = torch.nn.Parameter(torch.rand(1, 1, 10))
        self.num_mixtures = num_mixtures

    def get_q_x_given_obs(self, obs):
        obs_embed = self.embed_obs(obs)
        so_far_decoded = self.start.expand(obs_embed.shape)
        all_mu, all_sigma, all_mix_p = [], [], []
        for _ in range(self.num_mixtures):
            obs_dec = self.trans_dec(so_far_decoded, obs_embed)
            so_far_decoded = torch.cat([so_far_decoded, obs_dec[:, -1:, :]], dim=-2).detach()
            all_mu.append(self.decode_mu(obs_dec[:, -1:, :]))
            all_sigma.append(self.decode_sigma(obs_dec[:, -1:, :]))
            all_mix_p.append(self.decode_mixture_prob(obs_dec[:, -1:, :]))
        mixture_probs =  torch.cat(all_mix_p, dim=1).reshape(len(obs),self.num_mixtures)
        mu = torch.cat(all_mu, dim=1).reshape(len(obs),self.num_mixtures)
        sigma = torch.cat(all_sigma, dim=1).reshape(len(obs),self.num_mixtures)
        return BatchedNormalMixture1D(mixture_probs/mixture_probs.sum(dim=1).unsqueeze(-1), mu, sigma)

    def forward(self, theta, obs):
        q_x_given_obs = self.get_q_x_given_obs(obs)
        log_prob = q_x_given_obs.log_prob(theta)
        return -torch.mean(log_prob)


class SeqGaussMixPosteriorV2(nn.Module):
    def __init__(self, num_mixtures, d_model, num_obs_vars, num_trans_layers):
        super().__init__()
        self.embed_obs = MLP(input_size=num_obs_vars, output_size=d_model, num_hidden_units=d_model*2,
                             num_hidden_layers=2, unsqueeze=False)
        self.trans_enc = nn.Sequential(*[nn.TransformerEncoderLayer(d_model=d_model, nhead=d_model//10,
                                                                   dim_feedforward=d_model*2,
                                                                   dropout=0.0, batch_first=True)
                                         for _ in range(num_trans_layers)])
        self.trans_dec = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=d_model, nhead=d_model//10,
                                                                   dim_feedforward=d_model*2,
                                                                   dropout=0.0, batch_first=True),
                                               num_layers=num_trans_layers)

        self.decode_mu = MLP(input_size=d_model, num_hidden_layers=1, num_hidden_units=d_model*2,
                             unsqueeze=False)
        self.decode_sigma = MLP(input_size=d_model, num_hidden_layers=1, num_hidden_units=d_model*2,
                                last_activation=torch.exp,
                                unsqueeze=False)
        self.decode_mixture_prob = MLP(input_size=d_model, num_hidden_layers=1, num_hidden_units=d_model*2,
                                       last_activation=torch.exp,
                                       unsqueeze=False)

        self.start = torch.nn.Parameter(torch.rand(1, 1, d_model))
        self.num_mixtures = num_mixtures
        self.d_model = d_model

    def get_q_x_given_obs(self, obs):
        '''

        :param obs: shape: Batch, Seq, 1 (single feature)
        :return:
        '''
        memory = self.embed_obs(obs)
        memory = self.trans_enc(memory)
        so_far_decoded = self.start.expand(obs.shape[0],1, self.d_model)
        all_mu, all_sigma, all_mix_p = [], [], []
        for _ in range(self.num_mixtures):
            dec = self.trans_dec(so_far_decoded, memory)
            so_far_decoded = torch.cat([so_far_decoded, dec[:, -1:, :]], dim=-2).detach()
            all_mu.append(self.decode_mu(dec[:, -1:, :]))
            all_sigma.append(self.decode_sigma(dec[:, -1:, :]))
            all_mix_p.append(self.decode_mixture_prob(dec[:, -1:, :]))
        mixture_probs = torch.cat(all_mix_p, dim=1).reshape(len(obs), self.num_mixtures)
        mu = torch.cat(all_mu, dim=1).reshape(len(obs), self.num_mixtures)
        sigma = torch.cat(all_sigma, dim=1).reshape(len(obs), self.num_mixtures)
        return BatchedNormalMixture1D(mixture_probs / mixture_probs.sum(dim=1).unsqueeze(-1), mu, sigma)

    def forward(self, theta, obs):
        q_x_given_obs = self.get_q_x_given_obs(obs)
        log_prob = q_x_given_obs.log_prob(theta)
        return -torch.mean(log_prob)

def amortize_inference(inference_network, gen_traces_fn, optimizer, num_iterations):
    loss_history = []
    for i in range(num_iterations):
        x, obs = gen_traces_fn()
        loss = inference_network(x, obs)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_history.append(loss.item())

    return loss_history
