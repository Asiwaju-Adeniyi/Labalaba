#include <cuda_runtime.h>

/*
  One butterfly stage Bi.

  X_in, X_out: (B, F, L)
  W_stage:     (L, 2)   -- two weights per row
  stage:       0 ... log2(L)-1
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

    // idx -> (b, f, l)
    int l = idx % L;
    int tmp = idx / L;
    int f = tmp % F;
    int b = tmp / F;

    int stride  = 1 << stage;
    int partner = l ^ stride;

    if (partner >= L) return;

    // Enforce ONE thread per butterfly pair
    if (l < partner) {
        int base_l = b * F * L + f * L + l;
        int base_p = b * F * L + f * L + partner;

        float x0 = X_in[base_l];
        float x1 = X_in[base_p];

        // Load full 2×2 butterfly weights
        float a = W_stage[l * 2 + 0];
        float b0 = W_stage[l * 2 + 1];
        float c = W_stage[partner * 2 + 0];
        float d = W_stage[partner * 2 + 1];

        // Apply butterfly transform
        X_out[base_l] = a * x0 + b0 * x1;
        X_out[base_p] = c * x0 + d * x1;
    }
}
