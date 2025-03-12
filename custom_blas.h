#ifndef CUSTOM_BLAS_H
#define CUSTOM_BLAS_H

// Takes in a matrix in row-major A, and computes y = A * x.
void gemv(const double *A, const double *x, double *y, int m, int n);

#endif