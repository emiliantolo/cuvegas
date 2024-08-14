import torch
import numba
from numba import cuda
from vegas_cuda_extension.python_wrapper import integrate
import numpy as np
import math
import time

def func(x):
    n_dim = x.shape[-1]
    sigma2 = torch.tensor(0.0001)
    mu = torch.tensor(0.5)
    f = torch.sum(torch.square(x - mu), axis=1)
    return torch.exp(-0.5 * f / sigma2) / torch.pow(2.0 * np.pi * sigma2, n_dim / 2.0)

@cuda.jit(device=True, inline=True)
def device_func(x):
    sigma2 = 0.0001
    mu = 0.5
    f = 0.0
    for i in range(n_dim):
        f += (x[i] - mu) ** 2
    return math.exp(-0.5 * f / sigma2) / pow(2.0 * math.pi * sigma2, n_dim / 2.0)

n_dim = 4
max_it = 20
id = [[0, 1]] * n_dim

tot_eval = 1000000
n_intervals = 1024

start = time.time()
cuda_result = integrate(device_func, domain=id, n_intervals=n_intervals, max_it=max_it, n_dim=n_dim, n_evals=tot_eval)
cuda_elapsed = time.time() - start

print('res: {}\ntime: {}'.format(cuda_result, cuda_elapsed))
