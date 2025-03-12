#include "custom_blas.h"

void gemv(const double *A, const double *x, double *y, int m, int n)
{
    for (int i = 0; i < m; ++i)
    {
        double sum = 0.0;
        for (int j = 0; j < n; ++j)
        {
            sum += A[i * n + j] * x[j];
        }
        y[i] = sum;
    }
}