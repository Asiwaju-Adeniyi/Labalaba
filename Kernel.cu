#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>

__global__ void naive_butterfly_xw_kernel(
    float *X, float *W, float *Y,
    float *B, float *F, float *L, float *e)
{

    int bf = blockIdx.x;    

    int b = bf / F;         
    int f = bf % F;         

    int tid = threadIdx.x;  
}



