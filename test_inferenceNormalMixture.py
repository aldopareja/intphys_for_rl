import torch
torch.manual_seed(0)
from utils import NormalMixture1D
from main import InferenceNetworkPQ_NormalMixture
from matplotlib import pyplot as plt
import seaborn as sns

gen_model = NormalMixture1D([0.3,0.3,0.4], means=[-3, 2, 0], stds=[1,1,1])

def get_traces(gen_model, num_traces, obs_std):
    x = gen_model.sample(num_traces)
    obs = torch.distributions.Normal(loc=x,scale=obs_std).sample((1,)).squeeze()
    return x, obs

def print_grads(model):
    for p in model.parameters():
        print('grads')
        print(p.grad.max(), p.grad.min(), p.grad.mean())
        print('abs')
        print(p.max(), p.min(), p.mean())

def amortize_inference(inference_network, gen_traces_fn, optimizer, num_iterations):
    loss_history = []
    for i in range(num_iterations):

        x, obs = gen_traces_fn()
        loss = inference_network(x, obs)
        print(i, loss)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_history.append(loss.item())

    return loss_history

gen_traces_fn = lambda: get_traces(gen_model, 100, 1.0)
inference_network = InferenceNetworkPQ_NormalMixture(10,1,2,200)
optimizer = torch.optim.Adam(params=inference_network.parameters())


losses = amortize_inference(inference_network, gen_traces_fn, optimizer, 2000)