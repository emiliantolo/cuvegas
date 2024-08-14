#ifndef FUNCTIONS_CUH
#define FUNCTIONS_CUH

__host__ __device__ double linear(double* x, int n_dim);
__host__ __device__ double roosarnold(double* x, int n_dim);
__host__ __device__ double morokoffcalfisch(double* x, int n_dim);
__host__ __device__ double hellekalek(double* x, int n_dim);
__host__ __device__ double ridge(double* x, int n_dim);
__host__ __device__ double expon(double* x, int n_dim);
__host__ __device__ double cosin(double* x, int n_dim);
__host__ __device__ double dirichlet(double* x, int n_dim);
__host__ __device__ double gaussian(double* x, int n_dim);
__host__ __device__ double sinexp(double* x, int n_dim);
__host__ __device__ double one(double* x, int n_dim);

#endif