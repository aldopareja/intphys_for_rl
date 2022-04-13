import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, num_hidden_layers, num_hidden_units, input_size=1,
                 output_size=1, activation=nn.ReLU(), last_activation=lambda x: x):

        super().__init__()
        layers = [nn.Sequential(nn.Linear(input_size,num_hidden_units),activation)]
        layers += [nn.Sequential(nn.Linear(num_hidden_units, num_hidden_units), activation)
                   for _ in range(num_hidden_layers - 1)]
        layers += [nn.Sequential(nn.Linear(num_hidden_units, output_size))]
        self.network = nn.Sequential(*layers)
        self.last_activation = last_activation

    def forward(self,obs):
        return self.last_activation(self.network(obs.unsqueeze(-1)).squeeze())

class NormalMixture1D(torch.distributions.Distribution):
    def __init__(self, mixture_probs, means, stds):
        self.num_mixtures = len(mixture_probs)
        self.mixture_probs = torch.tensor(mixture_probs, dtype=torch.float32) if isinstance(mixture_probs, list) else mixture_probs
        self.normals = torch.distributions.Normal(
            loc=torch.tensor(means, dtype=torch.float32) if isinstance(means, list) else means,
            scale=torch.tensor(stds, dtype=torch.float32) if isinstance(stds, list) else stds,
        )

    def sample(self, n) -> torch.Tensor:
        mix_idx = torch.multinomial(self.mixture_probs, n, True).unsqueeze(-1)
        normals = self.normals.sample((n,))
        return torch.gather(normals, 1, mix_idx).squeeze()

    def log_prob(self, value):
        value_expanded = value.unsqueeze(-1).expand(-1, self.num_mixtures)
        normal_log_prob = self.normals.log_prob(value_expanded)
        category_log_prob = torch.log(self.mixture_probs)
        return torch.logsumexp(normal_log_prob + category_log_prob, dim=1)