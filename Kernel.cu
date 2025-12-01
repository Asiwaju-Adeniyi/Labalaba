#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>

__global__ void naive_butterfly_xw_kernel(
    float *X, float *W, float *Y,
    int B, int F, int L, int e)
{
    // Which row we’re working on
    int bf = blockIdx.x;
    int b = bf / F;
    int f = bf % F;

    int tid = threadIdx.x;  // lane inside the row

    // Pointer to the beginning of this (b,f) row
    int row_offset = (b * F + f) * L;
    float* x_row = X + row_offset;
    float* y_row = Y + row_offset; 

    extern __shared__ float buffer[];


}

