#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>
#include <torch/extension.h>

#define THREADS 256

extern "C" __global__
void sphkv_logit_kernel(
    const float*   __restrict__ q_all,          // [H_q, dh]
    const float*   __restrict__ cb_flat,         // [H_kv, num_tiers, G_max, cb_max, g_max]
    const int*     __restrict__ tier_G_arr,      // [num_tiers]
    const int*     __restrict__ tier_g_arr,      // [num_tiers]
    const uint8_t* __restrict__ theta_codes,     // [P, page_size, G_max]
    const uint8_t* __restrict__ radius_codes,    // [P, page_size, G_max]
    const float*   __restrict__ r_scales,        // [P, G_max]
    const int*     __restrict__ block_table,     // [H_kv, max_blocks]
    const int*     __restrict__ bits_table,      // [H_kv, max_blocks]
    const int*     __restrict__ ctx_lens,        // [H_kv]
    float*         __restrict__ logits_out,      // [H_q, max_ctx]
    int num_q_heads, int num_kv_heads, int kv_groups,
    int num_tiers, int max_blocks, int page_size, int dh,
    int G_max, int cb_max, int g_max,
    float sm_scale, int max_ctx
){
    const int hq  = blockIdx.x;
    const int tok = blockIdx.y * THREADS + threadIdx.x;
    const int hkv = hq / kv_groups;

    if (hq >= num_q_heads || tok >= ctx_lens[hkv]) {
        if (hq < num_q_heads && tok < max_ctx)
            logits_out[hq * max_ctx + tok] = -1e9f;
        return;
    }

    const int pg   = tok / page_size;
    const int slot = tok % page_size;
    const int phys = block_table[hkv * max_blocks + pg];
    const int tier = bits_table[hkv * max_blocks + pg];

    const int G_tier = tier_G_arr[tier];
    const int g_tier = tier_g_arr[tier];

    const int code_base   = (phys * page_size + slot) * G_max;
    const int rscale_base = phys * G_max;

    // Codebook strides: [H_kv, num_tiers, G_max, cb_max, g_max]
    const int cb_stride_hkv = num_tiers * G_max * cb_max * g_max;
    const int cb_stride_t   = G_max * cb_max * g_max;
    const int cb_stride_g   = cb_max * g_max;
    const int cb_stride_c   = g_max;

    const float* q_head  = q_all + hq * dh;
    const float* cb_tier = cb_flat + hkv * cb_stride_hkv + tier * cb_stride_t;

    float logit_sum = 0.f;
    for (int g = 0; g < G_tier; g++) {
        const int   code  = (int)theta_codes[code_base + g];
        const float r_c   = (float)radius_codes[code_base + g];
        const float r_hat = r_scales[rscale_base + g] * r_c / 255.0f;

        const float* q_g  = q_head + g * g_tier;
        const float* cb_g = cb_tier + g * cb_stride_g + code * cb_stride_c;

        // B.2 Step 1: compute ||q(g)||
        float q_norm_sq = 0.f;
        for (int d = 0; d < g_tier; d++) {
            q_norm_sq += q_g[d] * q_g[d];
        }
        float q_norm = sqrtf(q_norm_sq + 1e-12f);  // eps for stability

        // B.2 Step 2: dot(q_dir, codebook) = dot(q/||q||, cb)
        float dot = 0.f;
        for (int d = 0; d < g_tier; d++) {
            dot += q_g[d] * cb_g[d];
        }
        // Normalize: cos_theta = dot / ||q(g)||
        float cos_theta = dot / (q_norm + 1e-6f);

        // B.2 Step 3: cosine clipping
        cos_theta = fminf(fmaxf(cos_theta, -1.0f), 1.0f);

        // B.2 Step 4: logit += ||q(g)|| * r_hat * cos_theta
        logit_sum += q_norm * r_hat * cos_theta;
    }

    // Apply 1/sqrt(dh) scaling
    float logit = logit_sum * sm_scale;

    // Sentinel: empty slots have r_hat=0 everywhere → logit_sum=0
    logits_out[hq * max_ctx + tok] = (logit_sum == 0.0f) ? -1e9f : logit;
}


void sphkv_logit_launcher(
    torch::Tensor q_all,
    torch::Tensor cb_flat,
    torch::Tensor tier_G_arr,
    torch::Tensor tier_g_arr,
    torch::Tensor theta_codes,
    torch::Tensor radius_codes,
    torch::Tensor r_scales,
    torch::Tensor block_table,
    torch::Tensor bits_table,
    torch::Tensor ctx_lens,
    torch::Tensor logits_out,
    int num_q_heads, int num_kv_heads, int kv_groups,
    int num_tiers, int max_blocks, int page_size, int dh,
    int G_max, int cb_max, int g_max,
    float sm_scale, int max_ctx
){
    dim3 grid(num_q_heads, (max_ctx + THREADS - 1) / THREADS);
    dim3 block(THREADS);
    sphkv_logit_kernel<<<grid, block>>>(
        q_all.data_ptr<float>(),
        cb_flat.data_ptr<float>(),
        tier_G_arr.data_ptr<int>(),
        tier_g_arr.data_ptr<int>(),
        theta_codes.data_ptr<uint8_t>(),
        radius_codes.data_ptr<uint8_t>(),
        r_scales.data_ptr<float>(),
        block_table.data_ptr<int>(),
        bits_table.data_ptr<int>(),
        ctx_lens.data_ptr<int>(),
        logits_out.data_ptr<float>(),
        num_q_heads, num_kv_heads, kv_groups,
        num_tiers, max_blocks, page_size, dh,
        G_max, cb_max, g_max, sm_scale, max_ctx
    );
}
