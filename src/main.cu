#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cub/cub.cuh>
#include <omp.h>
#include "config.cuh"
#include "commons/common.cuh"

const double gpu_thr = 0.1;
const double multi_thr = 1e8;

__global__
void setup_kernel(curandState *state, long int it, long int offset) {
    long int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < it) { 
        curand_init(123, idx, offset, &state[idx]);
    }
}

template <int n_dim>
__global__ void vegasFill(int n_strat, int n_intervals, int n_edges, long int it, long int batch_size, int runs, int *counts, double *weights, double *x_edges, double *dx_edges, double *JF, double *JF2, int *evals, curandState *dev_states) {

    long int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if(idx < batch_size) { //eval?

        //state
        curandState state = dev_states[idx];

        int my_runs = (batch_size * (runs - 1) + idx) < it ? runs : (runs - 1);

        for (int run = 0; run < my_runs; run++) {

            //cube index
            int cub = evals[run * batch_size + idx];

            double y;
            int id[n_dim];
            double x[n_dim];

            //get y
            double dy = 1.0 / n_strat;
            int tmp = cub;
            for (int i = 0; i < n_dim; i++) {
                int q = tmp / n_strat;
                y = curand_uniform(&state) * dy + (tmp - (q * n_strat)) * dy;
                tmp = q;
                //get interval id, integer part of mapped point
                id[i] = (int) (y * n_intervals);
                id[i] = id[i] >= n_intervals ? n_intervals - 1 : id[i];
                //get x
                x[i] = x_edges[i*n_edges+id[i]] + dx_edges[i*n_intervals+id[i]] * ((y * (double) n_intervals) - id[i]);
            }

            //get jac
            double jac = 1.0;
            for (int i = 0; i < n_dim; i++) {
                jac *= n_intervals * dx_edges[i*n_intervals+id[i]];
            }

            double jf_t = integrand(x) * jac;
            double jf2_t = jf_t * jf_t;

            //accumulate weight
            for (int i = 0; i < n_dim; i++) {
                atomicAdd(&(weights[i*n_intervals+id[i]]), jf2_t);
                atomicAdd(&(counts[i*n_intervals+id[i]]), 1);
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

__global__
void results(int n_cubes, double v_cubes, double beta, long int *nh, double *JF, double *JF2, double *dh, double *sum_r, double *sum_s) {

    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if(i < n_cubes) {

        int neval = nh[i];

        double Ih = JF[i] / neval * v_cubes;
        double Sig2 = (JF2[i] / neval * v_cubes * v_cubes) - (Ih * Ih);

        sum_r[i] = Ih;
        sum_s[i] = Sig2 / neval;

        double dh_tmp = v_cubes * v_cubes * JF2[i] / neval - pow(v_cubes * JF[i] / neval, 2);

        if(dh_tmp < 0) {
            dh_tmp = 0;
        }

        dh[i] = pow(dh_tmp, beta);
    }
}

__global__
void normalizeWeights(int n_dim, int n_intervals, int *counts, double *weights) {

    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < (n_dim * n_intervals)) {

        //normalize weight
        if (counts[i] != 0) {
            weights[i] /= counts[i];
        }
    }
}

__global__
void smoothWeights(int n_dim, int n_intervals, double alpha, double *weights, double *smoothed_weights, double *d_sum) {

    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < (n_dim * n_intervals)) {

        int dim = i / n_intervals;
        int interval = i % n_intervals;

        double d_tmp;
        //smooth weight
        if (interval == 0) {
            d_tmp = (7.0 * weights[i] + weights[i+1]) / (8.0 * d_sum[dim]);
        } else if (interval == (n_intervals - 1)) {
            d_tmp = (weights[i-1] + 7.0 * weights[i]) / (8.0 * d_sum[dim]);
        } else {
            d_tmp = (weights[i-1] + 6.0 * weights[i] + weights[i+1]) / (8.0 * d_sum[dim]);
        }
        //smooth alpha
        if (d_tmp != 0) {
            d_tmp = pow((d_tmp - 1.0) / log(d_tmp), alpha);
        }
        smoothed_weights[i] = d_tmp;
    }
}

__global__
void resetWeights(int n_dim, int n_intervals, int *counts, double *weights) {

    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < (n_dim * n_intervals)) {

        //reset weights
        weights[i] = 0;
        counts[i] = 0;
    }
}

__global__
void resetWeightStrat(int n_cubes, double *JF, double *JF2) {

    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if(i < n_cubes) {

        //reset weight strat
        JF[i] = 0;
        JF2[i] = 0;
    }
}

__global__
void updateNh(int n_cubes, long int n_eval_it, long int *nh, double *dh, double *dh_sum, bool *hit) {

    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if(i < n_cubes) {

        // update nh
        double f = dh[i] / *dh_sum;
        //one should succeed
        if(f > gpu_thr) *hit = true;
        int nh_s = f * n_eval_it;
        nh[i] = nh_s < 2 ? 2 : nh_s;
    }
}

__global__
void setMap(int n_dim, int n_intervals, int *old_intervals, double *summed_weights, double *smoothed_weights_sum) {

    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < (n_dim * n_intervals)) {

        int dim = i / n_intervals;
        int interval = i % n_intervals;

        double delta_weights = summed_weights[dim] / n_intervals;
        int div = (int) (smoothed_weights_sum[i] / delta_weights);

        int off = interval > 0 ? (int) (smoothed_weights_sum[i-1] / delta_weights) : 0;
        int num = div - off;

        for (int j = 0; j < num; j++) {
            old_intervals[dim*n_intervals+off+j] = interval;
        }
    }
}

__global__
void updateXEdges(int n_dim, int n_intervals, int n_edges, int *old_intervals, double *summed_weights, double *smoothed_weights, double *smoothed_weights_sum, double *x_edges, double *x_edges_old, double *dx_edges_old) {

    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < (n_dim * (n_intervals - 1))) {

        int dim = i / (n_intervals - 1);
        int interval = i % (n_intervals - 1);

        double delta_weights = summed_weights[dim] / n_intervals;

        int new_interval = interval + 1;
        int old_interval = old_intervals[dim*n_intervals+interval];

        double acc = new_interval * delta_weights - (old_interval > 0 ? smoothed_weights_sum[dim*n_intervals+old_interval-1] : 0);

        x_edges[dim*n_edges+new_interval] = x_edges_old[dim*n_edges+old_interval] + acc / smoothed_weights[dim*n_intervals+old_interval] * dx_edges_old[dim*n_intervals+old_interval];
    }
}

__global__
void updateDxEdges(int n_dim, int n_intervals, int n_edges, double *x_edges, double *dx_edges) {

    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < (n_dim * n_intervals)) {

        int dim = i / n_intervals;
        int interval = i % n_intervals;

        int new_interval = interval + 1;

        dx_edges[dim*n_intervals+new_interval-1] = x_edges[dim*n_edges+new_interval] - x_edges[dim*n_edges+new_interval-1];
    }
}

__global__
void mapCubes(int n_cubes, int *evals, long int *nh, long int *nh_sum) {

    long int i = threadIdx.x + blockIdx.x * blockDim.x;

    if(i < n_cubes) {

        long int neval = nh[i];
        long int offset = nh_sum[i];

        for (int e = 0; e < neval; e++) {
            evals[offset+e] = i;
        }
    }
}

template<class T>
__global__
void sumVals(int n, T *a, T *b) {
    
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < n) {
        a[i] += b[i];
    }

}


int main() {

    double startTime = cpuMilliSeconds();

    // multi gpu
    int n_gpus;
    checkCudaError(cudaGetDeviceCount(&n_gpus));

    for (int i = 0; i < n_gpus; i++) {
        checkCudaError(cudaSetDevice(i));
        cudaFree(0);
    }

    double startTimeNocontext = cpuMilliSeconds();

    printf("GPUs: %i\n", n_gpus);
    checkCudaError(cudaSetDevice(0));

    for (int i = 0; i < n_gpus; i++) {
        for (int j = 0; j < n_gpus; j++) {
            if (i != j) {
            checkCudaError(cudaSetDevice(i));
            checkCudaError(cudaDeviceEnablePeerAccess(j, 0));
            }
        }
    }

    long int n_eval_it = tot_eval / max_it;

    bool multi_map = false;
    int nt;
    if (n_eval_it >= multi_thr) {
        //multi thread
        nt = omp_get_max_threads();
        printf("NT: %i\n", nt);
        omp_set_num_threads(nt);
        multi_map = true;
    }

    //vegas map
    int n_edges = n_intervals + 1;
    double *x_edges, **x_edges_d;
    double *dx_edges, **dx_edges_d;
    double **weights_d; //ndim,n_intervals
    int **counts_d; //ndim,n_intervals
    double *x_edges_old_d;
    double *dx_edges_old_d;
    double *smoothed_weights_d;
    double *smoothed_weights_sum_d;
    double *summed_weights_d;

    //vegas stratification
    int n_strat = (int) pow((n_eval_it / 4.0), 1.0 / n_dim);
    int n_cubes = pow(n_strat, n_dim);
    double v_cubes = pow((1.0 / n_strat), n_dim); 
    double *dh_d; //sample counts dampened
    long int *nh, *nh_d; //statified sample counts per cube
    double **JF_d;
    double **JF2_d;

    long int max_eval_it = n_eval_it + 2 * (long int) n_cubes;
    long int batch_size = max_eval_it < max_batch_size ? max_eval_it : max_batch_size;

    double *Results;
    double *Sigma2;

    double *res_s, *res_s_d;
    double *sig_s, *sig_s_d;
    double *dh_sum_d;
    double *d_sum_d;

    double *sum_r_d;
    double *sum_s_d;

    int *evals, **evals_d;
    long int *nh_sum, *nh_sum_d;

    int *old_intervals_d;

    curandState **dev_states;

    bool hit_gpu_limit = false, *hit_gpu_limit_d;

PUSH_RANGE("init", 1)

    x_edges = (double*) malloc(n_dim*n_edges * sizeof(double));
    dx_edges = (double*) malloc(n_dim*n_intervals * sizeof(double));

    nh = (long int*) malloc(n_cubes * sizeof(long int));
    //checkCudaError(cudaMallocHost(&nh, n_cubes*sizeof(long int)));

    Results = (double*) malloc(max_it * sizeof(double));
    Sigma2 = (double*) malloc(max_it * sizeof(double));

    res_s = (double*) malloc(sizeof(double));
    sig_s = (double*) malloc(sizeof(double));

    evals = (int*) malloc(max_eval_it * sizeof(int));
    //checkCudaError(cudaMallocHost(&evals, max_eval_it*sizeof(int)));
    nh_sum = (long int*) malloc(n_cubes * sizeof(long int));
    //checkCudaError(cudaMallocHost(&nh_sum, n_cubes*sizeof(long int)));

    checkCudaError(cudaMalloc(&x_edges_old_d, n_dim*n_edges*sizeof(double)));
    checkCudaError(cudaMalloc(&dx_edges_old_d, n_dim*n_intervals*sizeof(double)));
    checkCudaError(cudaMalloc(&smoothed_weights_d, n_dim*n_intervals*sizeof(double)));
    checkCudaError(cudaMalloc(&smoothed_weights_sum_d, n_dim*n_intervals*sizeof(double)));

    checkCudaError(cudaMalloc(&dh_d, n_cubes*sizeof(double)));
    checkCudaError(cudaMalloc(&nh_d, n_cubes*sizeof(long int)));
    checkCudaError(cudaMalloc(&res_s_d, sizeof(double)));
    checkCudaError(cudaMalloc(&sig_s_d, sizeof(double)));
    checkCudaError(cudaMalloc(&dh_sum_d, sizeof(double)));

    checkCudaError(cudaMalloc(&d_sum_d, n_dim*sizeof(double)));
    checkCudaError(cudaMalloc(&summed_weights_d, n_dim*sizeof(double)));

    checkCudaError(cudaMalloc(&sum_r_d, n_cubes*sizeof(double)));
    checkCudaError(cudaMalloc(&sum_s_d, n_cubes*sizeof(double)));

    checkCudaError(cudaMalloc(&nh_sum_d, n_cubes*sizeof(long int)));

    checkCudaError(cudaMalloc(&old_intervals_d, n_dim*n_intervals*sizeof(int)));

    checkCudaError(cudaMalloc(&hit_gpu_limit_d, sizeof(bool)));

    weights_d = (double**) malloc(n_gpus * sizeof(double*));
    counts_d = (int**) malloc(n_gpus * sizeof(int*));
    JF_d = (double**) malloc(n_gpus * sizeof(double*));
    JF2_d = (double**) malloc(n_gpus * sizeof(double*));
    evals_d = (int**) malloc(n_gpus * sizeof(int*));
    x_edges_d = (double**) malloc(n_gpus * sizeof(double*));
    dx_edges_d = (double**) malloc(n_gpus * sizeof(double*));
    for (int i = 0; i < n_gpus; i++) {
        checkCudaError(cudaSetDevice(i));    
        checkCudaError(cudaMalloc(&weights_d[i], n_dim*n_intervals*sizeof(double)));
        checkCudaError(cudaMalloc(&counts_d[i], n_dim*n_intervals*sizeof(int)));
        checkCudaError(cudaMalloc(&JF_d[i], n_cubes*sizeof(double)));
        checkCudaError(cudaMalloc(&JF2_d[i], n_cubes*sizeof(double)));
        checkCudaError(cudaMalloc(&evals_d[i], max_eval_it*sizeof(int)));
        checkCudaError(cudaMalloc(&x_edges_d[i], n_dim*n_edges*sizeof(double)));
        checkCudaError(cudaMalloc(&dx_edges_d[i], n_dim*n_intervals*sizeof(double)));
    }
    checkCudaError(cudaSetDevice(0));

    // cuda parameters
    int gridSizeRes = n_cubes / blockSizeRes + (int) ((n_cubes % blockSizeRes) != 0);
    int gridSizeUpd = n_cubes / blockSizeUpd + (int) ((n_cubes % blockSizeUpd) != 0);
    int gridSizeFill = batch_size / blockSizeFill + (int) ((batch_size % blockSizeFill) != 0);
    int gridSizeIntervals = n_dim * n_intervals / blockSizeIntervals + (int) (((n_dim * n_intervals) % blockSizeIntervals) != 0);    
    int gridSizeInit = batch_size / blockSizeInit + (int) ((batch_size % blockSizeInit) != 0);

    printf("Strat: %i\n", n_strat);
    printf("Cubes: %i\n", n_cubes);
    printf("Intervals: %i\n", n_intervals);

    // init dx_edges and x_edges
    double step = 1.0 / n_intervals;
    for (int i = 0; i < n_dim; i++) {
        for (int j = 0; j < n_edges; j++) {
            x_edges[i*n_edges+j] = j * step;
        }
        for (int j = 0; j < n_intervals; j++) {
            dx_edges[i*n_intervals+j] = x_edges[i*n_edges+j+1] - x_edges[i*n_edges+j];
        }
    }

    // init nh
    int nh_s = 1.0 / (double) n_cubes * n_eval_it;
    nh_s = nh_s < 2 ? 2 : nh_s;
    for (int i = 0; i < n_cubes; i++) {
        nh[i] = nh_s;
    }

    checkCudaError(cudaMemcpy(x_edges_d[0], x_edges, n_dim*n_edges*sizeof(double), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(dx_edges_d[0], dx_edges, n_dim*n_intervals*sizeof(double), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(nh_d, nh, n_cubes*sizeof(long int), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(hit_gpu_limit_d, &hit_gpu_limit, sizeof(bool), cudaMemcpyHostToDevice));

    //init
    dev_states = (curandState**) malloc(n_gpus * sizeof(curandState*));
    for (int i = 0; i < n_gpus; i++) {
        checkCudaError(cudaSetDevice(i));
        checkCudaError(cudaMalloc(&dev_states[i], batch_size*sizeof(curandState)));
    }

    long int offset = 0;
    for (int i = 0; i < n_gpus; i++) {
        checkCudaError(cudaSetDevice(i));
        setup_kernel<<<gridSizeInit,blockSizeInit>>>(dev_states[i], batch_size, offset);
        offset += (tot_eval / n_gpus);
    }

    for (int i = 0; i < n_gpus; i++) {
        checkCudaError(cudaSetDevice(i));
        checkCudaError(cudaPeekAtLastError());
        checkCudaError(cudaDeviceSynchronize());
    }
    checkCudaError(cudaSetDevice(0));

    double res = 0;
    double sigmas = 0;

    // priority need to check
    // get the range of stream priorities for this device
    int priority_high, priority_low;
    checkCudaError(cudaDeviceGetStreamPriorityRange(&priority_low, &priority_high));
    // create streams with highest and lowest available priorities
    cudaStream_t st_high, st_low1, st_low2;
    checkCudaError(cudaStreamCreateWithPriority(&st_high, cudaStreamNonBlocking, priority_high));
    checkCudaError(cudaStreamCreateWithPriority(&st_low1, cudaStreamNonBlocking, priority_low));
    checkCudaError(cudaStreamCreateWithPriority(&st_low2, cudaStreamNonBlocking, priority_low));

    // Determine temporary device storage requirements
    void *d_temp_storage_red_0 = NULL;
    size_t temp_storage_red_0_bytes = 0;
    void *d_temp_storage_red = NULL;
    size_t temp_storage_red_bytes = 0;
    void *d_temp_storage_sum = NULL;
    size_t temp_storage_sum_bytes = 0;
    void *d_temp_storage_red_int = NULL;
    size_t temp_storage_red_int_bytes = 0;
    void *d_temp_storage_sum_cub = NULL;
    size_t temp_storage_sum_cub_bytes = 0;
    cub::DeviceReduce::Sum(d_temp_storage_red_0, temp_storage_red_0_bytes, dh_d, dh_sum_d, n_cubes);
    cub::DeviceReduce::Sum(d_temp_storage_red, temp_storage_red_bytes, sum_r_d, res_s_d, n_cubes);
    cub::DeviceScan::InclusiveSum(d_temp_storage_sum, temp_storage_sum_bytes, smoothed_weights_d, smoothed_weights_sum_d, n_intervals);
    cub::DeviceReduce::Sum(d_temp_storage_red_int, temp_storage_red_int_bytes, weights_d[0], d_sum_d, n_intervals);
    cub::DeviceScan::ExclusiveSum(d_temp_storage_sum_cub, temp_storage_sum_cub_bytes, nh_d, nh_sum_d, n_cubes);

    // Allocate temporary storage
    checkCudaError(cudaMalloc(&d_temp_storage_red_0, temp_storage_red_bytes));
    checkCudaError(cudaMalloc(&d_temp_storage_red, temp_storage_red_bytes));
    checkCudaError(cudaMalloc(&d_temp_storage_sum, temp_storage_sum_bytes));
    checkCudaError(cudaMalloc(&d_temp_storage_red_int, temp_storage_red_int_bytes));
    checkCudaError(cudaMalloc(&d_temp_storage_sum_cub, temp_storage_sum_cub_bytes));

    int it = 0;
    long int tot_nevals = 0;

    double startTimeIt = cpuMilliSeconds();

POP_RANGE

    do {

PUSH_RANGE("map", 2)

        // reset weight strat
        for (int i = 0; i < n_gpus; i++) {
            checkCudaError(cudaSetDevice(i));
            resetWeightStrat<<<gridSizeRes,blockSizeRes>>>(n_cubes, JF_d[i], JF2_d[i]);
            resetWeights<<<gridSizeIntervals,blockSizeIntervals>>>(n_dim, n_intervals, counts_d[i], weights_d[i]);
        }

        int a = 1;
        while (a < n_gpus) {
            for (int i = 0; i < a; i++) {
                if ((i + a) < n_gpus) {
                    checkCudaError(cudaSetDevice(i));
                    checkCudaError(cudaMemcpyAsync(x_edges_d[i + a], x_edges_d[i], n_dim*n_edges*sizeof(double), cudaMemcpyDeviceToDevice));
                    checkCudaError(cudaMemcpyAsync(dx_edges_d[i + a], dx_edges_d[i], n_dim*n_intervals*sizeof(double), cudaMemcpyDeviceToDevice));
                }
            }
            a *= 2;
        }

        checkCudaError(cudaSetDevice(0));

        long int idx = 0;

        checkCudaError(cudaMemcpy(&hit_gpu_limit, hit_gpu_limit_d, sizeof(bool), cudaMemcpyDeviceToHost));

        //map evaluations to cubes
        if (hit_gpu_limit) {
            //cpu
            if (multi_map) {
                //multi
                checkCudaError(cudaMemcpyAsync(nh, nh_d, n_cubes*sizeof(long int), cudaMemcpyDeviceToHost));
                cub::DeviceScan::ExclusiveSum(d_temp_storage_sum_cub, temp_storage_sum_cub_bytes, nh_d, nh_sum_d, n_cubes);
                checkCudaError(cudaMemcpyAsync(nh_sum, nh_sum_d, n_cubes*sizeof(long int), cudaMemcpyDeviceToHost));
                checkCudaError(cudaPeekAtLastError());
                checkCudaError(cudaDeviceSynchronize());
                int i, e;
                #pragma omp parallel for //schedule(guided) shared(evals) private(e) num_threads(16)
                for (i = 0; i < n_cubes; i++) {
                    for (e = 0; e < nh[i]; e++) {
                        evals[nh_sum[i]+e] = i;
                    }
                }
                idx = nh_sum[n_cubes - 1] + nh[n_cubes - 1];
            } else {
                //single
                checkCudaError(cudaMemcpy(nh, nh_d, n_cubes*sizeof(long int), cudaMemcpyDeviceToHost));
                for (int i = 0; i < n_cubes; i++) {
                    for (int e = 0; e < nh[i]; e++) {
                        evals[idx+e] = i;
                    }
                    idx += nh[i];
                }
            }
        } else {
            //gpu
            cub::DeviceScan::ExclusiveSum(d_temp_storage_sum_cub, temp_storage_sum_cub_bytes, nh_d, nh_sum_d, n_cubes);
            checkCudaError(cudaDeviceSynchronize());
            mapCubes<<<gridSizeRes,blockSizeRes>>>(n_cubes, evals_d[0], nh_d, nh_sum_d);
            checkCudaError(cudaPeekAtLastError());
            checkCudaError(cudaDeviceSynchronize());
            long int last_nh;
            long int last_nh_sum;
            checkCudaError(cudaMemcpy(&last_nh, &(nh_d[n_cubes - 1]), sizeof(long int), cudaMemcpyDeviceToHost));
            checkCudaError(cudaMemcpy(&last_nh_sum, &(nh_sum_d[n_cubes - 1]), sizeof(long int), cudaMemcpyDeviceToHost));
            idx = last_nh_sum + last_nh;
        }

        long int offset = 0;
        long int jump = idx / n_gpus + (int) ((idx % n_gpus) != 0);
        long int ex;
        //#pragma omp parallel
        for (int i = 0; i < n_gpus; i++) {
            offset = i * jump;
            ex = (offset + jump) <= idx ? jump : (idx - offset);
            checkCudaError(cudaSetDevice(i));
            if (hit_gpu_limit) {
                checkCudaError(cudaMemcpyAsync(evals_d[i], &(evals[offset]), ex*sizeof(int), cudaMemcpyHostToDevice));
            } else {
                if (i != 0) {
                    checkCudaError(cudaMemcpyAsync(evals_d[i], &((evals_d[0])[offset]), ex*sizeof(int), cudaMemcpyDeviceToDevice));
                }
            }
        }

        hit_gpu_limit = false;
        checkCudaError(cudaMemcpy(hit_gpu_limit_d, &hit_gpu_limit, sizeof(bool), cudaMemcpyHostToDevice));

        tot_nevals += idx;

POP_RANGE

        //call kernel
PUSH_RANGE("fill", 3)

        int runs;
        //#pragma omp parallel
        for (int i = 0; i < n_gpus; i++) {
            offset = i * jump;
            ex = (offset + jump) <= idx ? jump : (idx - offset);
            runs = ex / batch_size + (int) ((ex % batch_size) != 0);
            checkCudaError(cudaSetDevice(i));
            checkCudaError(cudaDeviceSynchronize());
            //call kernel
            vegasFill<n_dim><<<gridSizeFill,blockSizeFill>>>(n_strat, n_intervals, n_edges, ex, batch_size, runs, counts_d[i], weights_d[i], x_edges_d[i], dx_edges_d[i], JF_d[i], JF2_d[i], evals_d[i], dev_states[i]);
        }

        for (int i = 0; i < n_gpus; i++) {
            checkCudaError(cudaSetDevice(i));
            checkCudaError(cudaPeekAtLastError());
            checkCudaError(cudaDeviceSynchronize());
        }

        a = 1;
        while (a < n_gpus) {
            for (int i = 0; i < n_gpus; i += (2 * a)) {
                if ((i + a) < n_gpus) {
                    checkCudaError(cudaSetDevice(i));
                    sumVals<double><<<gridSizeRes,blockSizeRes>>>(n_cubes, JF_d[i], JF_d[i + a]);
                    sumVals<double><<<gridSizeRes,blockSizeRes>>>(n_cubes, JF2_d[i], JF2_d[i + a]);
                    sumVals<double><<<gridSizeIntervals,blockSizeIntervals>>>(n_dim*n_intervals, weights_d[i], weights_d[i + a]);
                    sumVals<int><<<gridSizeIntervals,blockSizeIntervals>>>(n_dim*n_intervals, counts_d[i], counts_d[i + a]);
                }
            }
            for (int i = 0; i < n_gpus; i += (2 * a)) {
                if ((i + a) < n_gpus) {
                    checkCudaError(cudaSetDevice(i));
                    checkCudaError(cudaDeviceSynchronize());
                }
            }
            a *= 2;
        }

        checkCudaError(cudaSetDevice(0));

POP_RANGE

PUSH_RANGE("remaining", 4)

        results<<<gridSizeRes,blockSizeRes>>>(n_cubes, v_cubes, beta, nh_d, JF_d[0], JF2_d[0], dh_d, sum_r_d, sum_s_d);

        checkCudaError(cudaPeekAtLastError());
        checkCudaError(cudaDeviceSynchronize());

        normalizeWeights<<<gridSizeIntervals,blockSizeIntervals,0,st_high>>>(n_dim, n_intervals, counts_d[0], weights_d[0]);
        for (int i = 0; i < n_dim; i++) {
            cub::DeviceReduce::Sum(d_temp_storage_red_int, temp_storage_red_int_bytes, &(weights_d[0][i*n_intervals]), &(d_sum_d[i]), n_intervals, st_high);
        }

        smoothWeights<<<gridSizeIntervals,blockSizeIntervals,0,st_high>>>(n_dim, n_intervals, alpha, weights_d[0], smoothed_weights_d, d_sum_d);
        for (int i = 0; i < n_dim; i++) {
            cub::DeviceReduce::Sum(d_temp_storage_red_int, temp_storage_red_int_bytes, &(smoothed_weights_d[i*n_intervals]), &(summed_weights_d[i]), n_intervals, st_high);
        }

        cub::DeviceReduce::Sum(d_temp_storage_red_0, temp_storage_red_0_bytes, dh_d, dh_sum_d, n_cubes, st_low1);
        updateNh<<<gridSizeUpd,blockSizeUpd,0,st_low1>>>(n_cubes, n_eval_it, nh_d, dh_d, dh_sum_d, hit_gpu_limit_d);

        if (it >= skip) {
            cub::DeviceReduce::Sum(d_temp_storage_red, temp_storage_red_bytes, sum_r_d, res_s_d, n_cubes, st_low2);
            cub::DeviceReduce::Sum(d_temp_storage_red, temp_storage_red_bytes, sum_s_d, sig_s_d, n_cubes, st_low2);
        }

        checkCudaError(cudaMemcpyAsync(x_edges_old_d, x_edges_d[0], n_dim*n_edges*sizeof(double), cudaMemcpyDeviceToDevice, st_high));
        checkCudaError(cudaMemcpyAsync(dx_edges_old_d, dx_edges_d[0], n_dim*n_intervals*sizeof(double), cudaMemcpyDeviceToDevice, st_high));

        for (int i = 0; i < n_dim; i++) {
            cub::DeviceScan::InclusiveSum(d_temp_storage_sum, temp_storage_sum_bytes, &(smoothed_weights_d[i*n_intervals]), &(smoothed_weights_sum_d[i*n_intervals]), n_intervals, st_high);
        }

        setMap<<<gridSizeIntervals,blockSizeIntervals,0,st_high>>>(n_dim, n_intervals, old_intervals_d, summed_weights_d, smoothed_weights_sum_d);
        updateXEdges<<<gridSizeIntervals,blockSizeIntervals,0,st_high>>>(n_dim, n_intervals, n_edges, old_intervals_d, summed_weights_d, smoothed_weights_d, smoothed_weights_sum_d, x_edges_d[0], x_edges_old_d, dx_edges_old_d);
        updateDxEdges<<<gridSizeIntervals,blockSizeIntervals,0,st_high>>>(n_dim, n_intervals, n_edges, x_edges_d[0], dx_edges_d[0]);

        it++;

        if (it > skip) {
            checkCudaError(cudaMemcpyAsync(res_s, res_s_d, sizeof(double), cudaMemcpyDeviceToHost, st_low2));
            checkCudaError(cudaMemcpyAsync(sig_s, sig_s_d, sizeof(double), cudaMemcpyDeviceToHost, st_low2));
            checkCudaError(cudaStreamSynchronize(st_low2));

            Results[it - skip - 1] = *res_s;
            Sigma2[it - skip - 1] = *sig_s;

            //results
            res = 0;
            sigmas = 0;
            for (int i = 0; i < it - skip; i++) {
                res += Results[i] / Sigma2[i];
                sigmas += 1.0 / Sigma2[i];
            }
            res /= sigmas;
        }

        checkCudaError(cudaPeekAtLastError());
        checkCudaError(cudaDeviceSynchronize());

POP_RANGE

    } while(it < max_it);

    double elapsedTimeIt = cpuMilliSeconds() - startTimeIt;

PUSH_RANGE("clear", 1)

    //memory
    free(x_edges);
    free(dx_edges);
    free(Results);
    free(Sigma2);
    free(res_s);
    free(sig_s);

    //checkCudaError(cudaFreeHost(nh));
    //checkCudaError(cudaFreeHost(evals));
    //checkCudaError(cudaFreeHost(nh_sum));

    free(nh);
    free(evals);
    free(nh_sum);

    checkCudaError(cudaFree(x_edges_old_d));
    checkCudaError(cudaFree(dx_edges_old_d));
    checkCudaError(cudaFree(smoothed_weights_d));
    checkCudaError(cudaFree(smoothed_weights_sum_d));
    checkCudaError(cudaFree(dh_d));
    checkCudaError(cudaFree(nh_d));
    checkCudaError(cudaFree(res_s_d));
    checkCudaError(cudaFree(sig_s_d));
    checkCudaError(cudaFree(dh_sum_d));
    checkCudaError(cudaFree(d_sum_d));
    checkCudaError(cudaFree(summed_weights_d));
    checkCudaError(cudaFree(sum_r_d));
    checkCudaError(cudaFree(sum_s_d));
    checkCudaError(cudaFree(nh_sum_d));
    checkCudaError(cudaFree(old_intervals_d));
    checkCudaError(cudaFree(hit_gpu_limit_d));

    checkCudaError(cudaFree(d_temp_storage_red_0));
    checkCudaError(cudaFree(d_temp_storage_red));
    checkCudaError(cudaFree(d_temp_storage_sum));
    checkCudaError(cudaFree(d_temp_storage_red_int));
    checkCudaError(cudaFree(d_temp_storage_sum_cub));

    for (int i = 0; i < n_gpus; i++) {
        checkCudaError(cudaSetDevice(i));
        checkCudaError(cudaFree(dev_states[i]));
        checkCudaError(cudaFree(JF_d[i]));
        checkCudaError(cudaFree(JF2_d[i]));
        checkCudaError(cudaFree(weights_d[i]));
        checkCudaError(cudaFree(counts_d[i]));
        checkCudaError(cudaFree(evals_d[i]));
        checkCudaError(cudaFree(x_edges_d[i]));
        checkCudaError(cudaFree(dx_edges_d[i]));
    }
    free(dev_states);
    free(JF_d);
    free(JF2_d);
    free(weights_d);
    free(counts_d);
    free(evals_d);
    free(x_edges_d);
    free(dx_edges_d);

POP_RANGE

    double elapsedTime = cpuMilliSeconds() - startTime;
    double elapsedTimeNocontext = cpuMilliSeconds() - startTimeNocontext;

    printf("Result: %.12f\n", res);
    printf("Error: %.12f\n", 1.0 / sqrt(sigmas));
    printf("Total evals: %ld\n", tot_nevals);
    printf("Time elapsed %f ms\n", elapsedTime);
    printf("Iteration avg time %f ms\n", elapsedTimeIt / max_it);
    printf("Nocontext %f ms\n", elapsedTimeNocontext);
    printf("Context creation %f ms\n", elapsedTime - elapsedTimeNocontext);

    return(0);
}
