#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include "config.cuh"

extern "C" __device__ int integrand(double *return_value, double *x);

__device__ __forceinline__ int no_bank_conflict_index(int thread_id, int logical_index) {
    return logical_index * blockSizeFill + thread_id;
}

extern __shared__ int array[];
extern "C" __global__ void vegasFill(int n_dim, int n_strat, int n_intervals, int n_edges, long int it, long int batch_size, int runs, int *counts, double *weights, double *x_edges, double *dx_edges, double *JF, double *JF2, int *evals, curandState *dev_states) {

    long int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if(idx < batch_size) { //eval?

        //state
        curandState state = dev_states[idx];

        int my_runs = (batch_size * (runs - 1) + idx) < it ? runs : (runs - 1);

        int *id = (int*)array;
        double *x = &((double*)&id[blockSizeFill * n_dim])[threadIdx.x * n_dim];
        double y;

        for (int run = 0; run < my_runs; run++) {

            //cube index
            int cub = evals[run * batch_size + idx];

            //get y
            double dy = 1.0 / n_strat;
            int tmp = cub;
            for (int i = 0; i < n_dim; i++) {
                int q = tmp / n_strat;
                y = curand_uniform(&state) * dy + (tmp - (q * n_strat)) * dy;
                tmp = q;
                //get interval id, integer part of mapped point
                int i_idx = no_bank_conflict_index(threadIdx.x, i);
                id[i_idx] = (int) (y * n_intervals);
                id[i_idx] = id[i_idx] >= n_intervals ? n_intervals - 1 : id[i_idx];
                //get x
                x[i] = x_edges[i*n_edges+id[i_idx]] + dx_edges[i*n_intervals+id[i_idx]] * ((y * (double) n_intervals) - id[i_idx]);
            }

            //get jac
            double jac = 1.0;
            for (int i = 0; i < n_dim; i++) {
                jac *= n_intervals * dx_edges[i*n_intervals+id[no_bank_conflict_index(threadIdx.x, i)]];
            }

            double res;
            integrand(&res, x);
            double jf_t = res * jac;
            double jf2_t = jf_t * jf_t;

            //accumulate weight
            for (int i = 0; i < n_dim; i++) {
                int i_idx = no_bank_conflict_index(threadIdx.x, i);
                atomicAdd(&(weights[i*n_intervals+id[i_idx]]), jf2_t);
                atomicAdd(&(counts[i*n_intervals+id[i_idx]]), 1);
            }

            //accumulate weight strat
            //accumulate += jf and jf2
            atomicAdd(&(JF[cub]), jf_t);
            atomicAdd(&(JF2[cub]), jf2_t);
        }

        //state
        dev_states[idx] = state;
    }
}
