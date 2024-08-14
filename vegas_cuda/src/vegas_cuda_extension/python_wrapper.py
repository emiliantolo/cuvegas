import torch
import numba
from numba import cuda
import numpy as np
import os
import math
from vegas_cuda_extension.cuda_extension import integrate_cuda

def integrate(device_function,n_dim,domain=None,n_evals=1000000,max_it=20,skip=5,batch_size=1048576,n_intervals=1024,n_strat=None,alpha=0.5,beta=0.75,seed=None,multi_gpu=None,multi_cpu=None,gpu_num=-1,cpu_num=-1,gpu_thr=0.1):

    #set seed
    if seed == None:
        seed = int.from_bytes(os.urandom(4), 'big')

    #set n_strat
    if n_strat == None:
        n_eval_it = n_evals // max_it
        n_strat = int(math.pow((n_eval_it / 4.0), 1.0 / n_dim))

    #check domain
    if domain != None:
        #remap
        dstart = np.zeros(n_dim, dtype=np.float64)
        drange = np.zeros(n_dim, dtype=np.float64)
        r = 1
        for i in range(n_dim):
            a = domain[i][0]
            b = domain[i][1]
            dstart[i] = (a)
            drange[i] = (b - a)
            r *= drange[i]
        if (not np.any(dstart)) and (np.isin(drange, [1]).all()):
            #copy
            device_func = device_function
        else:
            @cuda.jit(device=True, inline=True)
            def device_func(x):
                d_dstart = cuda.const.array_like(dstart)
                d_drange = cuda.const.array_like(drange)
                for i in range(n_dim):
                    x[i] = (x[i] * d_drange[i] + d_dstart[i])
                res = device_function(x)
                return r * res
    else:
        #copy
        device_func = device_function

    #compile
    ptx_code = numba.cuda.compile_ptx_for_current_device(device_func, numba.float64(numba.types.CPointer(numba.float64)), device=True)[0]

    #change name
    for l in ptx_code.split('\n'):
        if '// .globl' in l:
            name = l.split('// .globl')[1].strip() 
    if name == None:
        print('Not found')
        exit()
    else:
        ptx_code = ptx_code.replace(name, 'integrand')

    data_path = os.path.join(os.path.dirname(__file__), 'cuda_kernel.ptx')

    #multi gpu
    if multi_gpu == None:
        multi_gpu = n_evals >= 1e10

    #multi cpu
    if multi_cpu == None:
        multi_cpu = (n_evals // max_it) >= 1e8

    #execute
    res = integrate_cuda(n_evals, max_it, skip, batch_size, n_intervals, n_strat, n_dim, alpha, beta, seed, multi_gpu, multi_cpu, gpu_num, cpu_num, gpu_thr, str(ptx_code), str(data_path))

    #build result tuple
    res = (res[0], {'error': res[1], 'n_strat': res[2], 'n_cubes': res[3], 'n_intervals': res[4], 'tot_nevals': res[5], 'tot_time': res[6], 'avg_it_time': res[7], 'seed': res[8], 'multi_gpu': res[9], 'multi_cpu': res[10], 'gpu_num': res[11], 'omp_threads_num': res[12], 'gpu_thr': res[13]})

    return res
