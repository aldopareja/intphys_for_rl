import torch
torch.manual_seed(0)
import  utils as u
import main as m

ts = torch.tensor([0.0,1.0,2,3,4,5,8,10,15,20,30,40,60,70,80])
n_vars = 1
batch_size = 1000
d_model = 30
noise_std_ratio = 0.1

inference_network = m.SeqGaussMixPosteriorV2(num_mixtures=2, d_model=d_model,
                                             num_obs_vars=n_vars,
                                             num_trans_layers=1).cuda()
optimizer = torch.optim.Adam(params=inference_network.parameters())

gen_traces_fn = lambda: m.solve_ode_sample(batch_size, n_vars, [-10.0, 10.0], [-0.1, 0.1], noise_std_ratio, ts,
                                          device='cuda')

losses = m.amortize_inference(inference_network, gen_traces_fn, optimizer, 2000)