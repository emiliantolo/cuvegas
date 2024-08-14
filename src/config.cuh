#ifndef CONFIG_CUH
#define CONFIG_CUH
#include "commons/functions.cuh"
#define _USE_MATH_DEFINES
#include <math.h>

//vegas settings
long int tot_eval = 10000000;
int max_it = 20;
int skip = 5;
long int max_batch_size = 1048576;

//vegas map
const int n_intervals = 1024;
double alpha = 0.5;

//vegas stratification
double beta = 0.75;

//blocks
const int blockSizeRes = 64;
const int blockSizeRed = 128;
const int blockSizeUpd = 128;
const int blockSizeFill = 64;
const int blockSizeIntervals = 128;
const int blockSizeInit = 128;

//integrand
const int n_dim = 4;
__host__ __device__
double integrand(double* x) {
    return ridge(x, n_dim);
}

#endif