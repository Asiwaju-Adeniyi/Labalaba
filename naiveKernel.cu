#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>

__global__ void butterfly_factor_kernel(
    const float* X,
    const float* B,
    float* Y,
    int K, int a, int b, int c, int d)
{
    int i = blockIdx.x;
    int j = blockIdx.y;

    int col_base = (i * (c * d)) + j;

    int row_base = (i * (b * d)) + j;

    int ℓ = threadIdx.x;
    if (ℓ >= c) return;

    int col = col_base + ℓ * d;

    for (int k = 0; k < b; k++) {
        int row = row_base + k * d;

        float weight = B[col * (b*d) + row];

        for (int r = 0; r < K; r++) {
            int in_idx  = r * (c*d*a) + col;
            int out_idx = r * (b*d*a) + row;
            Y[out_idx] += X[in_idx] * weight;
        }
    }
}

