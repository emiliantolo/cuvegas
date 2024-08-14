#define _USE_MATH_DEFINES
#include <math.h>
#include "functions.cuh"

__host__ __device__
double linear(double* x, int n_dim) {
    double f = 0.0;
    for (int i = 0; i < n_dim; i++) {
        f += x[i];
    }
    return f;
}

__host__ __device__
double roosarnold(double* x, int n_dim) {
	double f = 1.0/(double)n_dim;
	double aux = 0.0;
	for (int i = 0; i < n_dim; i++){
		aux += fabsf(4.0 * x[i] - 2.0);
	}
	f *= aux;
	return f;
}

__host__ __device__
double morokoffcalfisch(double* x, int n_dim) {
    double f = 1.0;
    for (int i = 0; i < n_dim; i++) {
        f *= pow(x[i], 1.0 / (double) n_dim);
    }
    f *= pow(1.0 + 1.0 / (double) n_dim, n_dim);
    return f;  
}

__host__ __device__
double hellekalek(double* x, int n_dim)
{
	double f = 1.0;
	for (int i = 0; i < n_dim; i++){
		f *= ((x[i] - 0.5)/sqrtf(12.0));
	}
	return f;
}

__host__ __device__
double ridge(double* x, int n_dim) {
	int N = 1000;
    double norm = pow(100.0 / M_PI, 2.0) / N;
    double f = 0.0;
	for (int i = 0; i < N; i++){
        double x0 = i / (N - 1.0);
        double dx2 = 0.0;
        for (int j = 0; j < n_dim; j++){
            double a = (x[j] - x0);
            dx2 += a * a;
        }
		f += exp(-100.0 * dx2);
	}
	return f * norm;
}

__host__ __device__
double expon(double* x, int n_dim) {
    double f = 0.0;
    for (int i = 0; i < n_dim; i++){
		f += x[i] * x[i];
	}
    return exp(-f);
}

__host__ __device__
double cosin(double* x, int n_dim) {
    double f = 1.0;
    for (int i = 0; i < n_dim; i++){
		f *= cos(x[i]);
	}
    return f;
}

__host__ __device__
double dirichlet(double* x, int n_dim) {
    double alpha = 5.0;
    double f = 1.0;
    double b = pow(tgamma(alpha), n_dim) / tgamma(alpha * n_dim);
    for (int i = 0; i < n_dim; i++){
		f *= pow(x[i], alpha - 1.0);
	}
    return f / b;
}

__host__ __device__
double gaussian(double* x, int n_dim) {
	double sigma2 = 0.0001;
    double mu = 0.5;
    double f = 0.0;
    for (int i = 0; i < n_dim; i++){
        double a = x[i] - mu;
		f += a * a;
	}
	return exp(-0.5 * f / sigma2) / pow(2.0 * M_PI * sigma2, n_dim / 2.0);
}

__host__ __device__
double sinexp(double* x, int n_dim) {
    return sin(x[0]) + exp(x[1]);
}

__host__ __device__
double one(double* x, int n_dim) {
    return 1.0;
}
