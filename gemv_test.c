#include <stdio.h>
#include <time.h>
#include "custom_blas.h"
#include "cblas.h"

int main()
{
    int Nmax = 8192; // Largest matrix to test.
    int maxpow = 13; // Nmax = 2 ^ maxpow
    int nreps = 50;  // Number of times to repeat each gemv.

    double *A = malloc(sizeof(double) * Nmax * Nmax);
    double *x = malloc(sizeof(double) * Nmax);
    double *y = malloc(sizeof(double) * Nmax);

    // Fill data matrix and vector.
    for (int i = 0; i < Nmax; ++i)
    {
        x[i] = i;
        for (int j = 0; j < Nmax; ++j)
        {
            A[i * Nmax + j] = i * Nmax + j;
        }
    }

    double custom_gflops[maxpow];
    double blas_gflops[maxpow];
    int size = 2;
    int idx = 0;
    struct timespec start_time, end_time;
    double elapsed_time;
    while (size <= Nmax)
    {
        // Number of floating point operations.
        int num_ops = 2 * size * size - size;
        double custom_time = 1e10;
        for (int i = 0; i < nreps; ++i)
        {
            clock_gettime(CLOCK_MONOTONIC, &start_time);
            gemv(A, x, y, size, size);
            clock_gettime(CLOCK_MONOTONIC, &end_time);
            elapsed_time = (end_time.tv_sec - start_time.tv_sec) * 1e9 +
                           (end_time.tv_nsec - start_time.tv_nsec);
            custom_time = MIN(elapsed_time, custom_time);
        }
        custom_gflops[idx] = num_ops / custom_time;
        printf("Custom GFLOPs = %f ", num_ops / custom_time);

        double blas_time = 1e10;
        for (int i = 0; i < nreps; ++i)
        {
            clock_gettime(CLOCK_MONOTONIC, &start_time);
            cblas_dgemv(CblasRowMajor, CblasNoTrans, size, size, 1.0, A, size, x, 1, 0.0, y, 1);
            clock_gettime(CLOCK_MONOTONIC, &end_time);
            elapsed_time = (end_time.tv_sec - start_time.tv_sec) * 1e9 +
                           (end_time.tv_nsec - start_time.tv_nsec);
            blas_time = MIN(elapsed_time, blas_time);
        }
        blas_gflops[idx++] = num_ops / blas_time;
        printf("BLAS GFLOPs = %f ", num_ops / blas_time);
        printf("\n\n");
        size *= 2;
    }

    FILE *file = fopen("gemv_results.csv", "w");
    for (int i = 0; i < maxpow; ++i)
    {
        fprintf(file, "%f, ", custom_gflops[i]);
    }
    fprintf(file, "\n");
    for (int i = 0; i < maxpow; ++i)
    {
        fprintf(file, "%f, ", blas_gflops[i]);
    }
}