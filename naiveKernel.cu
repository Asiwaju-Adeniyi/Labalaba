#include <cuda_runtime.h>
#include <cmath>

/*
  One butterfly stage.

  X_in, X_out: (B, F, L)
  W_stage:     (L, 2)  -- two weights per output position
  stage:       which butterfly level (0 ... log2(L)-1)
*/

__global__ void butterfly_stage_kernel(
    const float* X_in,
    float* X_out,
    const float* W_stage,
    int B, int F, int L,
    int stage
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * F * L;

    if (idx >= total) return;

    // Flattened index → (b, f, l)
    int l = idx % L;
    int tmp = idx / L;
    int f = tmp % F;
    int b = tmp / F;

    int stride = 1 << stage;               // 2^stage
    int partner = l ^ stride;              // butterfly pairing

    if (partner >= L) return;

    // Read inputs
    float x0 = X_in[b * F * L + f * L + l];
    float x1 = X_in[b * F * L + f * L + partner];

    // Two weights per output position
    float w0 = W_stage[l * 2 + 0];
    float w1 = W_stage[l * 2 + 1];

    // Butterfly combine
    X_out[b * F * L + f * L + l] = w0 * x0 + w1 * x1;
}
