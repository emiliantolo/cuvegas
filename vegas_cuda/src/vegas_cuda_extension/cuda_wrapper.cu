#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cub/cub.cuh>
#include <omp.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include "config.cuh"
#include "commons/common.cuh"

// helper functions and utilities to work with CUDA
#define CUDA_DRIVER_API
#include <helper_cuda.h>
#include <helper_cuda_drvapi.h>
#include <helper_functions.h>  // helper for shared that are common to CUDA Samples

__global__
void setup_kernel(curandState *state, long int it, long int offset, unsigned int seed) {
    long int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < it) { 
        curand_init(seed, idx, offset, &state[idx]);
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
void updateNh(int n_cubes, long int n_eval_it, long int *nh, double *dh, double *dh_sum, bool *hit, double gpu_thr) {

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

void jit(CUcontext *cuda_contexts, CUfunction *cuda_kernels, CUmodule *cuda_modules, const std::string& ptx_code, const std::string& path, int n_gpus) {

    CUlinkState cuLinkState;
    CUjit_option options[6];
    void *optionVals[6];
    float walltime;
    char error_log[8192], info_log[8192];
    unsigned int logSize = 8192;
    void *cuOut;
    size_t outSize;
    int myErr = 0;
    int myErr1 = 0;

    // Setup linker options
    // Return walltime from JIT compilation
    options[0] = CU_JIT_WALL_TIME;
    optionVals[0] = (void *) &walltime;
    // Pass a buffer for info messages
    options[1] = CU_JIT_INFO_LOG_BUFFER;
    optionVals[1] = (void *) info_log;
    // Pass the size of the info buffer
    options[2] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
    optionVals[2] = (void *) (long)logSize;
    // Pass a buffer for error message
    options[3] = CU_JIT_ERROR_LOG_BUFFER;
    optionVals[3] = (void *) error_log;
    // Pass the size of the error buffer
    options[4] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
    optionVals[4] = (void *) (long) logSize;
    // Make the linker verbose
    options[5] = CU_JIT_LOG_VERBOSE;
    optionVals[5] = (void *) 1;

    // Create a pending linker invocation
    checkCudaErrors(cuLinkCreate(6, options, optionVals, &cuLinkState));

    // Load ptx from file
    myErr = cuLinkAddFile(cuLinkState, CU_JIT_INPUT_PTX, path.c_str(), 0, 0, 0);
    if (myErr != CUDA_SUCCESS){
        // Errors will be put in error_log, per CU_JIT_ERROR_LOG_BUFFER option above.
        fprintf(stderr,"PTX Linker Error kernel:\n%s\n",error_log);
    }

    // Load ptx from string
    myErr1 = cuLinkAddData(cuLinkState, CU_JIT_INPUT_PTX, (void *)ptx_code.c_str(), strlen(ptx_code.c_str()) + 1, 0, 0, 0, 0);
    if (myErr1 != CUDA_SUCCESS){
        // Errors will be put in error_log, per CU_JIT_ERROR_LOG_BUFFER option above.
        fprintf(stderr,"PTX Linker Error function:\n%s\n",error_log);
    }

    // Complete the linker step
    checkCudaErrors(cuLinkComplete(cuLinkState, &cuOut, &outSize));

    // Linker walltime and info_log were requested in options above.
    //printf("CUDA Link Completed in %fms. Linker Output:\n%s\n", walltime, info_log);

    for (int i = 0; i < n_gpus; i++) {
        checkCudaErrors(cudaSetDevice(i));
        cuCtxGetCurrent(&(cuda_contexts[i]));
        checkCudaErrors(cuModuleLoadData(&(cuda_modules[i]), cuOut));
        checkCudaErrors(cuModuleGetFunction(&(cuda_kernels[i]), cuda_modules[i], "vegasFill"));
    }

    // Destroy the linker invocation
    checkCudaErrors(cuLinkDestroy(cuLinkState));
}

bool peer_enabled = false;

std::tuple<double,double,int,int,int,long int,double,double,unsigned int,bool,bool,int,int,double> integrate(long int tot_eval, int max_it, int skip, long int max_batch_size, int n_intervals, int n_strat, int n_dim, double alpha, double beta, unsigned int seed, bool multi_gpu, bool multi_cpu, int gpu_num, int cpu_num, double gpu_thr, const std::string& ptx_code, const std::string& path) {

    double startTime = cpuMilliSeconds();

    // multi gpu
    int n_gpus;

    if (multi_gpu) {
        checkCudaErrors(cudaGetDeviceCount(&n_gpus));
        //printf("GPUs: %i\n", n_gpus);

        if ((gpu_num > 0) && (gpu_num < n_gpus)) n_gpus = gpu_num;

        for (int i = 0; i < n_gpus; i++) {
            checkCudaErrors(cudaSetDevice(i));
            cudaFree(0);
        }

        if(!peer_enabled) {
            for (int i = 0; i < n_gpus; i++) {
                for (int j = 0; j < n_gpus; j++) {
                    if (i != j) {
                    checkCudaErrors(cudaSetDevice(i));
                    checkCudaErrors(cudaDeviceEnablePeerAccess(j, 0));
                    }
                }
            }
            peer_enabled = true;
        }
    } else {
        n_gpus = 1;
    }

    checkCudaError(cudaSetDevice(0));

    CUcontext *cuda_contexts = (CUcontext*) malloc(n_gpus * sizeof(CUcontext));
    CUmodule *cuda_modules = (CUmodule*) malloc(n_gpus * sizeof(CUmodule));
    CUfunction *cuda_kernels = (CUfunction*) malloc(n_gpus * sizeof(CUfunction));
    jit(cuda_contexts, cuda_kernels, cuda_modules, ptx_code, path, n_gpus);
    checkCudaErrors(cudaSetDevice(0));

    long int n_eval_it = tot_eval / max_it;

    //double gpu_thr = 0.1;
    //double multi_thr = 1e8;

    bool multi_map = false;
    int nt = NULL;

    if (multi_cpu) {
        //multi thread
        nt = omp_get_max_threads();
        //printf("NT: %i\n", nt);
        if ((cpu_num > 0) && (cpu_num < nt)) nt = cpu_num;
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
    //int n_strat = (int) pow((n_eval_it / 4.0), 1.0 / n_dim);
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
    //checkCudaErrors(cudaMallocHost(&nh, n_cubes*sizeof(long int)));

    Results = (double*) malloc(max_it * sizeof(double));
    Sigma2 = (double*) malloc(max_it * sizeof(double));

    res_s = (double*) malloc(sizeof(double));
    sig_s = (double*) malloc(sizeof(double));

    evals = (int*) malloc(max_eval_it * sizeof(int));
    //checkCudaErrors(cudaMallocHost(&evals, max_eval_it*sizeof(int)));
    nh_sum = (long int*) malloc(n_cubes * sizeof(long int));
    //checkCudaErrors(cudaMallocHost(&nh_sum, n_cubes*sizeof(long int)));

    checkCudaErrors(cudaMalloc(&x_edges_old_d, n_dim*n_edges*sizeof(double)));
    checkCudaErrors(cudaMalloc(&dx_edges_old_d, n_dim*n_intervals*sizeof(double)));
    checkCudaErrors(cudaMalloc(&smoothed_weights_d, n_dim*n_intervals*sizeof(double)));
    checkCudaErrors(cudaMalloc(&smoothed_weights_sum_d, n_dim*n_intervals*sizeof(double)));

    checkCudaErrors(cudaMalloc(&dh_d, n_cubes*sizeof(double)));
    checkCudaErrors(cudaMalloc(&nh_d, n_cubes*sizeof(long int)));
    checkCudaErrors(cudaMalloc(&res_s_d, sizeof(double)));
    checkCudaErrors(cudaMalloc(&sig_s_d, sizeof(double)));
    checkCudaErrors(cudaMalloc(&dh_sum_d, sizeof(double)));

    checkCudaErrors(cudaMalloc(&d_sum_d, n_dim*sizeof(double)));
    checkCudaErrors(cudaMalloc(&summed_weights_d, n_dim*sizeof(double)));

    checkCudaErrors(cudaMalloc(&sum_r_d, n_cubes*sizeof(double)));
    checkCudaErrors(cudaMalloc(&sum_s_d, n_cubes*sizeof(double)));

    checkCudaErrors(cudaMalloc(&nh_sum_d, n_cubes*sizeof(long int)));

    checkCudaErrors(cudaMalloc(&old_intervals_d, n_dim*n_intervals*sizeof(int)));

    checkCudaErrors(cudaMalloc(&hit_gpu_limit_d, sizeof(bool)));

    weights_d = (double**) malloc(n_gpus * sizeof(double*));
    counts_d = (int**) malloc(n_gpus * sizeof(int*));
    JF_d = (double**) malloc(n_gpus * sizeof(double*));
    JF2_d = (double**) malloc(n_gpus * sizeof(double*));
    evals_d = (int**) malloc(n_gpus * sizeof(int*));
    x_edges_d = (double**) malloc(n_gpus * sizeof(double*));
    dx_edges_d = (double**) malloc(n_gpus * sizeof(double*));
    for (int i = 0; i < n_gpus; i++) {
        checkCudaErrors(cudaSetDevice(i));    
        checkCudaErrors(cudaMalloc(&weights_d[i], n_dim*n_intervals*sizeof(double)));
        checkCudaErrors(cudaMalloc(&counts_d[i], n_dim*n_intervals*sizeof(int)));
        checkCudaErrors(cudaMalloc(&JF_d[i], n_cubes*sizeof(double)));
        checkCudaErrors(cudaMalloc(&JF2_d[i], n_cubes*sizeof(double)));
        checkCudaErrors(cudaMalloc(&evals_d[i], max_eval_it*sizeof(int)));
        checkCudaErrors(cudaMalloc(&x_edges_d[i], n_dim*n_edges*sizeof(double)));
        checkCudaErrors(cudaMalloc(&dx_edges_d[i], n_dim*n_intervals*sizeof(double)));
    }
    checkCudaErrors(cudaSetDevice(0));

    // cuda parameters
    int gridSizeRes = n_cubes / blockSizeRes + (int) ((n_cubes % blockSizeRes) != 0);
    int gridSizeUpd = n_cubes / blockSizeUpd + (int) ((n_cubes % blockSizeUpd) != 0);
    int gridSizeFill = batch_size / blockSizeFill + (int) ((batch_size % blockSizeFill) != 0);
    int gridSizeIntervals = n_dim * n_intervals / blockSizeIntervals + (int) (((n_dim * n_intervals) % blockSizeIntervals) != 0);    
    int gridSizeInit = batch_size / blockSizeInit + (int) ((batch_size % blockSizeInit) != 0);

    //printf("Strat: %i\n", n_strat);
    //printf("Cubes: %i\n", n_cubes);
    //printf("Intervals: %i\n", n_intervals);

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

    checkCudaErrors(cudaMemcpy(x_edges_d[0], x_edges, n_dim*n_edges*sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dx_edges_d[0], dx_edges, n_dim*n_intervals*sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(nh_d, nh, n_cubes*sizeof(long int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(hit_gpu_limit_d, &hit_gpu_limit, sizeof(bool), cudaMemcpyHostToDevice));

    //init
    dev_states = (curandState**) malloc(n_gpus * sizeof(curandState*));
    for (int i = 0; i < n_gpus; i++) {
        checkCudaErrors(cudaSetDevice(i));
        checkCudaErrors(cudaMalloc(&dev_states[i], batch_size*sizeof(curandState)));
    }

    long int offset = 0;
    for (int i = 0; i < n_gpus; i++) {
        checkCudaErrors(cudaSetDevice(i));
        setup_kernel<<<gridSizeInit,blockSizeInit>>>(dev_states[i], batch_size, offset, seed);
        offset += (tot_eval / n_gpus);
    }

    for (int i = 0; i < n_gpus; i++) {
        checkCudaErrors(cudaSetDevice(i));
        checkCudaErrors(cudaPeekAtLastError());
        checkCudaErrors(cudaDeviceSynchronize());
    }
    checkCudaErrors(cudaSetDevice(0));

    double res = 0;
    double sigmas = 0;

    // priority need to check
    // get the range of stream priorities for this device
    int priority_high, priority_low;
    checkCudaErrors(cudaDeviceGetStreamPriorityRange(&priority_low, &priority_high));
    // create streams with highest and lowest available priorities
    cudaStream_t st_high, st_low1, st_low2;
    checkCudaErrors(cudaStreamCreateWithPriority(&st_high, cudaStreamNonBlocking, priority_high));
    checkCudaErrors(cudaStreamCreateWithPriority(&st_low1, cudaStreamNonBlocking, priority_low));
    checkCudaErrors(cudaStreamCreateWithPriority(&st_low2, cudaStreamNonBlocking, priority_low));

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
    cub::DeviceReduce::Sum(d_temp_storage_red, temp_storage_red_bytes, dh_d, dh_sum_d, n_cubes);
    cub::DeviceScan::InclusiveSum(d_temp_storage_sum, temp_storage_sum_bytes, smoothed_weights_d, smoothed_weights_sum_d, n_intervals);
    cub::DeviceReduce::Sum(d_temp_storage_red_int, temp_storage_red_int_bytes, weights_d[0], d_sum_d, n_intervals);
    cub::DeviceScan::ExclusiveSum(d_temp_storage_sum_cub, temp_storage_sum_cub_bytes, nh_d, nh_sum_d, n_cubes);

    // Allocate temporary storage
    checkCudaErrors(cudaMalloc(&d_temp_storage_red_0, temp_storage_red_bytes));
    checkCudaErrors(cudaMalloc(&d_temp_storage_red, temp_storage_red_bytes));
    checkCudaErrors(cudaMalloc(&d_temp_storage_sum, temp_storage_sum_bytes));
    checkCudaErrors(cudaMalloc(&d_temp_storage_red_int, temp_storage_red_int_bytes));
    checkCudaErrors(cudaMalloc(&d_temp_storage_sum_cub, temp_storage_sum_cub_bytes));

    int it = 0;
    long int tot_nevals = 0;

    void ***args = (void***) malloc(n_gpus * sizeof(void**));
    long int *ex = (long int*) malloc(n_gpus * sizeof(long int));
    int *runs = (int*) malloc(n_gpus * sizeof(int));

    for (int i = 0; i < n_gpus; i++) {
        args[i] = (void**) malloc(15 * sizeof(void*));
        args[i][0] = (void*)&n_dim;
        args[i][1] = (void*)&n_strat;
        args[i][2] = (void*)&n_intervals;
        args[i][3] = (void*)&n_edges;
        args[i][4] = (void*)&(ex[i]);
        args[i][5] = (void*)&batch_size;
        args[i][6] = (void*)&(runs[i]);
        args[i][7] = (void*)&(counts_d[i]);
        args[i][8] = (void*)&(weights_d[i]); 
        args[i][9] = (void*)&(x_edges_d[i]);
        args[i][10] = (void*)&(dx_edges_d[i]);
        args[i][11] = (void*)&(JF_d[i]);
        args[i][12] = (void*)&(JF2_d[i]);
        args[i][13] = (void*)&(evals_d[i]);
        args[i][14] = (void*)&(dev_states[i]);
    }

    double startTimeIt = cpuMilliSeconds();

POP_RANGE

    do {

PUSH_RANGE("map", 2)

        // reset weight strat
        for (int i = 0; i < n_gpus; i++) {
            checkCudaErrors(cudaSetDevice(i));
            resetWeightStrat<<<gridSizeRes,blockSizeRes>>>(n_cubes, JF_d[i], JF2_d[i]);
            resetWeights<<<gridSizeIntervals,blockSizeIntervals>>>(n_dim, n_intervals, counts_d[i], weights_d[i]);
        }

        int a = 1;
        while (a < n_gpus) {
            for (int i = 0; i < a; i++) {
                if ((i + a) < n_gpus) {
                    checkCudaErrors(cudaSetDevice(i));
                    checkCudaErrors(cudaMemcpyAsync(x_edges_d[i + a], x_edges_d[i], n_dim*n_edges*sizeof(double), cudaMemcpyDeviceToDevice));
                    checkCudaErrors(cudaMemcpyAsync(dx_edges_d[i + a], dx_edges_d[i], n_dim*n_intervals*sizeof(double), cudaMemcpyDeviceToDevice));
                }
            }
            a *= 2;
        }

        checkCudaErrors(cudaSetDevice(0));

        long int idx = 0;

        checkCudaErrors(cudaMemcpy(&hit_gpu_limit, hit_gpu_limit_d, sizeof(bool), cudaMemcpyDeviceToHost));

        //map evaluations to cubes
        if (hit_gpu_limit) {
            //cpu
            if (multi_map) {
                //multi
                checkCudaErrors(cudaMemcpyAsync(nh, nh_d, n_cubes*sizeof(long int), cudaMemcpyDeviceToHost));
                cub::DeviceScan::ExclusiveSum(d_temp_storage_sum_cub, temp_storage_sum_cub_bytes, nh_d, nh_sum_d, n_cubes);
                checkCudaErrors(cudaMemcpyAsync(nh_sum, nh_sum_d, n_cubes*sizeof(long int), cudaMemcpyDeviceToHost));
                checkCudaErrors(cudaPeekAtLastError());
                checkCudaErrors(cudaDeviceSynchronize());
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
                checkCudaErrors(cudaMemcpy(nh, nh_d, n_cubes*sizeof(long int), cudaMemcpyDeviceToHost));
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
            checkCudaErrors(cudaDeviceSynchronize());
            mapCubes<<<gridSizeRes,blockSizeRes>>>(n_cubes, evals_d[0], nh_d, nh_sum_d);
            checkCudaErrors(cudaPeekAtLastError());
            checkCudaErrors(cudaDeviceSynchronize());
            long int last_nh;
            long int last_nh_sum;
            checkCudaErrors(cudaMemcpy(&last_nh, &(nh_d[n_cubes - 1]), sizeof(long int), cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(&last_nh_sum, &(nh_sum_d[n_cubes - 1]), sizeof(long int), cudaMemcpyDeviceToHost));
            idx = last_nh_sum + last_nh;
        }

        long int offset = 0;
        long int jump = idx / n_gpus + (int) ((idx % n_gpus) != 0);
        //#pragma omp parallel
        for (int i = 0; i < n_gpus; i++) {
            offset = i * jump;
            ex[i] = (offset + jump) <= idx ? jump : (idx - offset);
            checkCudaErrors(cudaSetDevice(i));
            if (hit_gpu_limit) {
                checkCudaErrors(cudaMemcpyAsync(evals_d[i], &(evals[offset]), ex[i]*sizeof(int), cudaMemcpyHostToDevice));
            } else {
                if (i != 0) {
                    checkCudaErrors(cudaMemcpyAsync(evals_d[i], &((evals_d[0])[offset]), ex[i]*sizeof(int), cudaMemcpyDeviceToDevice));
                }
            }
        }

        hit_gpu_limit = false;
        checkCudaErrors(cudaMemcpy(hit_gpu_limit_d, &hit_gpu_limit, sizeof(bool), cudaMemcpyHostToDevice));

        tot_nevals += idx;

POP_RANGE

        //call kernel
PUSH_RANGE("fill", 3)

        //#pragma omp parallel
        for (int i = 0; i < n_gpus; i++) {
            runs[i] = ex[i] / batch_size + (int) ((ex[i] % batch_size) != 0);
            checkCudaErrors(cudaSetDevice(i));
            checkCudaErrors(cudaDeviceSynchronize());
            //call kernel
            checkCudaErrors(cuLaunchKernel(cuda_kernels[i], gridSizeFill, 1, 1, blockSizeFill, 1, 1, blockSizeFill*n_dim*(sizeof(int)+sizeof(double)), nullptr, args[i], nullptr));
        }

        for (int i = 0; i < n_gpus; i++) {
            checkCudaErrors(cudaSetDevice(i));
            checkCudaErrors(cuCtxSynchronize());
            checkCudaErrors(cudaPeekAtLastError());
            checkCudaErrors(cudaDeviceSynchronize());
        }

        a = 1;
        while (a < n_gpus) {
            for (int i = 0; i < n_gpus; i += (2 * a)) {
                if ((i + a) < n_gpus) {
                    checkCudaErrors(cudaSetDevice(i));
                    sumVals<double><<<gridSizeRes,blockSizeRes>>>(n_cubes, JF_d[i], JF_d[i + a]);
                    sumVals<double><<<gridSizeRes,blockSizeRes>>>(n_cubes, JF2_d[i], JF2_d[i + a]);
                    sumVals<double><<<gridSizeIntervals,blockSizeIntervals>>>(n_dim*n_intervals, weights_d[i], weights_d[i + a]);
                    sumVals<int><<<gridSizeIntervals,blockSizeIntervals>>>(n_dim*n_intervals, counts_d[i], counts_d[i + a]);
                }
            }
            for (int i = 0; i < n_gpus; i += (2 * a)) {
                if ((i + a) < n_gpus) {
                    checkCudaErrors(cudaSetDevice(i));
                    checkCudaErrors(cudaDeviceSynchronize());
                }
            }
            a *= 2;
        }

        checkCudaErrors(cudaSetDevice(0));

POP_RANGE

PUSH_RANGE("remaining", 4)

        results<<<gridSizeRes,blockSizeRes>>>(n_cubes, v_cubes, beta, nh_d, JF_d[0], JF2_d[0], dh_d, sum_r_d, sum_s_d);

        checkCudaErrors(cudaPeekAtLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        normalizeWeights<<<gridSizeIntervals,blockSizeIntervals,0,st_high>>>(n_dim, n_intervals, counts_d[0], weights_d[0]);
        for (int i = 0; i < n_dim; i++) {
            cub::DeviceReduce::Sum(d_temp_storage_red_int, temp_storage_red_int_bytes, &(weights_d[0][i*n_intervals]), &(d_sum_d[i]), n_intervals, st_high);
        }

        smoothWeights<<<gridSizeIntervals,blockSizeIntervals,0,st_high>>>(n_dim, n_intervals, alpha, weights_d[0], smoothed_weights_d, d_sum_d);
        for (int i = 0; i < n_dim; i++) {
            cub::DeviceReduce::Sum(d_temp_storage_red_int, temp_storage_red_int_bytes, &(smoothed_weights_d[i*n_intervals]), &(summed_weights_d[i]), n_intervals, st_high);
        }

        cub::DeviceReduce::Sum(d_temp_storage_red_0, temp_storage_red_0_bytes, dh_d, dh_sum_d, n_cubes, st_low1);
        updateNh<<<gridSizeUpd,blockSizeUpd,0,st_low1>>>(n_cubes, n_eval_it, nh_d, dh_d, dh_sum_d, hit_gpu_limit_d, gpu_thr);

        if (it >= skip) {
            cub::DeviceReduce::Sum(d_temp_storage_red, temp_storage_red_bytes, sum_r_d, res_s_d, n_cubes, st_low2);
            cub::DeviceReduce::Sum(d_temp_storage_red, temp_storage_red_bytes, sum_s_d, sig_s_d, n_cubes, st_low2);
        }

        checkCudaErrors(cudaMemcpyAsync(x_edges_old_d, x_edges_d[0], n_dim*n_edges*sizeof(double), cudaMemcpyDeviceToDevice, st_high));
        checkCudaErrors(cudaMemcpyAsync(dx_edges_old_d, dx_edges_d[0], n_dim*n_intervals*sizeof(double), cudaMemcpyDeviceToDevice, st_high));

        for (int i = 0; i < n_dim; i++) {
            cub::DeviceScan::InclusiveSum(d_temp_storage_sum, temp_storage_sum_bytes, &(smoothed_weights_d[i*n_intervals]), &(smoothed_weights_sum_d[i*n_intervals]), n_intervals, st_high);
        }

        setMap<<<gridSizeIntervals,blockSizeIntervals,0,st_high>>>(n_dim, n_intervals, old_intervals_d, summed_weights_d, smoothed_weights_sum_d);
        updateXEdges<<<gridSizeIntervals,blockSizeIntervals,0,st_high>>>(n_dim, n_intervals, n_edges, old_intervals_d, summed_weights_d, smoothed_weights_d, smoothed_weights_sum_d, x_edges_d[0], x_edges_old_d, dx_edges_old_d);
        updateDxEdges<<<gridSizeIntervals,blockSizeIntervals,0,st_high>>>(n_dim, n_intervals, n_edges, x_edges_d[0], dx_edges_d[0]);

        it++;

        if (it > skip) {
            checkCudaErrors(cudaMemcpyAsync(res_s, res_s_d, sizeof(double), cudaMemcpyDeviceToHost, st_low2));
            checkCudaErrors(cudaMemcpyAsync(sig_s, sig_s_d, sizeof(double), cudaMemcpyDeviceToHost, st_low2));
            checkCudaErrors(cudaStreamSynchronize(st_low2));

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

        checkCudaErrors(cudaPeekAtLastError());
        checkCudaErrors(cudaDeviceSynchronize());

POP_RANGE

    } while(it < max_it);

    double elapsedTimeIt = cpuMilliSeconds() - startTimeIt;

PUSH_RANGE("clear", 1)

    for (int i = 0; i < n_gpus; i++) {
        checkCudaErrors(cuModuleUnload(cuda_modules[i]));
    }

    //memory
    free(x_edges);
    free(dx_edges);
    free(Results);
    free(Sigma2);
    free(res_s);
    free(sig_s);

    free(cuda_contexts);
    free(cuda_modules);
    free(cuda_kernels);

    free(ex);
    free(runs);

    for (int i = 0; i < n_gpus; i++) {
        free(args[i]);
    }
    free(args);

    //checkCudaErrors(cudaFreeHost(nh));
    //checkCudaErrors(cudaFreeHost(evals));
    //checkCudaErrors(cudaFreeHost(nh_sum));

    free(nh);
    free(evals);
    free(nh_sum);

    checkCudaErrors(cudaFree(x_edges_old_d));
    checkCudaErrors(cudaFree(dx_edges_old_d));
    checkCudaErrors(cudaFree(smoothed_weights_d));
    checkCudaErrors(cudaFree(smoothed_weights_sum_d));
    checkCudaErrors(cudaFree(dh_d));
    checkCudaErrors(cudaFree(nh_d));
    checkCudaErrors(cudaFree(res_s_d));
    checkCudaErrors(cudaFree(sig_s_d));
    checkCudaErrors(cudaFree(dh_sum_d));
    checkCudaErrors(cudaFree(d_sum_d));
    checkCudaErrors(cudaFree(summed_weights_d));
    checkCudaErrors(cudaFree(sum_r_d));
    checkCudaErrors(cudaFree(sum_s_d));
    checkCudaErrors(cudaFree(nh_sum_d));
    checkCudaErrors(cudaFree(old_intervals_d));
    checkCudaErrors(cudaFree(hit_gpu_limit_d));

    checkCudaErrors(cudaFree(d_temp_storage_red));
    checkCudaErrors(cudaFree(d_temp_storage_sum));
    checkCudaErrors(cudaFree(d_temp_storage_red_int));
    checkCudaErrors(cudaFree(d_temp_storage_sum_cub));

    for (int i = 0; i < n_gpus; i++) {
        checkCudaErrors(cudaSetDevice(i));
        checkCudaErrors(cudaFree(dev_states[i]));
        checkCudaErrors(cudaFree(JF_d[i]));
        checkCudaErrors(cudaFree(JF2_d[i]));
        checkCudaErrors(cudaFree(weights_d[i]));
        checkCudaErrors(cudaFree(counts_d[i]));
        checkCudaErrors(cudaFree(evals_d[i]));
        checkCudaErrors(cudaFree(x_edges_d[i]));
        checkCudaErrors(cudaFree(dx_edges_d[i]));
    }
    free(dev_states);
    free(JF_d);
    free(JF2_d);
    free(weights_d);
    free(counts_d);
    free(evals_d);
    free(x_edges_d);
    free(dx_edges_d);

    /*
    for (int i = 0; i < n_gpus; i++) {
        checkCudaErrors(cuCtxDestroy(cuda_contexts[i]));
    }
    */

POP_RANGE

    double elapsedTime = cpuMilliSeconds() - startTime;

    //printf("Result: %.8f\n", res);
    //printf("Error: %.8f\n", 1.0 / sqrt(sigmas));
    //printf("Total evals: %ld\n", tot_nevals);
    //printf("Time elapsed %f ms\n", elapsedTime);
    //printf("Iteration avg time %f ms\n", elapsedTimeIt / max_it);

    std::tuple ret = std::make_tuple(res, 1.0 / sqrt(sigmas), n_strat, n_cubes, n_intervals, tot_nevals, elapsedTime, elapsedTimeIt / max_it, seed, multi_gpu, multi_cpu, n_gpus, nt, gpu_thr);

    return(ret);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("integrate_cuda", &integrate, "CUDA wrapper function");
}
