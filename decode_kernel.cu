#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <math.h>
#include <torch/extension.h>

#define PAGE_THREADS 128

__global__
void sphkv_encode_append_kernel(
    const float*   __restrict__ k_post,
    const float*   __restrict__ v_new,
    const float*   __restrict__ decode_cb,
    uint8_t*       __restrict__ theta_codes,
    uint8_t*       __restrict__ radius_codes,
    float*         __restrict__ r_scales,
    half*          __restrict__ v_pool,
    const long*    __restrict__ pids,
    int slot, int G, int g, int C,
    int G_max, int dh, int page_size,
    int is_new_page
){
    const int h = blockIdx.x;
    const int tid = threadIdx.x;
    const int pid = (int)pids[h];
    const int v_base = (pid * page_size + slot) * dh;
    const int code_base = (pid * page_size + slot) * G_max;

    if (tid < dh) {
        v_pool[v_base + tid] = __float2half(v_new[h * dh + tid]);
    }

    if (tid < G) {
        const int grp = tid;
        const float* k_g = k_post + h * dh + grp * g;

        float norm_sq = 0.f;
        for (int d = 0; d < g; d++)
            norm_sq += k_g[d] * k_g[d];
        float norm = sqrtf(norm_sq + 1e-16f);
        float inv_norm = 1.0f / (norm + 1e-8f);

        const float* cb = decode_cb + (h * G * C + grp * C) * g;
        float best_sim = -1e30f;
        int best_code = 0;
        for (int c = 0; c < C; c++) {
            float sim = 0.f;
            for (int d = 0; d < g; d++)
                sim += k_g[d] * inv_norm * cb[c * g + d];
            if (sim > best_sim) {
                best_sim = sim;
                best_code = c;
            }
        }

        theta_codes[code_base + grp] = (uint8_t)best_code;

        float r_scale;
        if (is_new_page) {
            r_scales[pid * G_max + grp] = norm;
            r_scale = norm;
        } else {
            r_scale = r_scales[pid * G_max + grp];
        }
        float r_q = roundf(norm / (r_scale + 1e-8f) * 255.0f);
        r_q = fminf(fmaxf(r_q, 0.f), 255.f);
        radius_codes[code_base + grp] = (uint8_t)r_q;
    }
}


void sphkv_encode_append_launcher(
    torch::Tensor k_post, torch::Tensor v_new,
    torch::Tensor decode_cb,
    torch::Tensor theta_codes, torch::Tensor radius_codes,
    torch::Tensor r_scales, torch::Tensor v_pool,
    torch::Tensor pids,
    int slot, int G, int g, int C,
    int G_max, int dh, int page_size, int is_new_page
){
    const int num_kv = k_post.size(0);
    dim3 grid(num_kv);
    dim3 block(PAGE_THREADS);
    sphkv_encode_append_kernel<<<grid, block>>>(
        k_post.data_ptr<float>(), v_new.data_ptr<float>(),
        decode_cb.data_ptr<float>(),
        theta_codes.data_ptr<uint8_t>(), radius_codes.data_ptr<uint8_t>(),
        r_scales.data_ptr<float>(),
        (half*)v_pool.data_ptr(),
        pids.data_ptr<long>(),
        slot, G, g, C, G_max, dh, page_size, is_new_page);
}


__global__
void sphkv_fused_kernel(
    const float*   __restrict__ q_all,
    const float*   __restrict__ cb_flat,
    const int*     __restrict__ tier_G_arr,
    const int*     __restrict__ tier_g_arr,
    const uint8_t* __restrict__ theta_codes,
    const uint8_t* __restrict__ radius_codes,
    const float*   __restrict__ r_scales,
    const half*    __restrict__ v_pool,
    const int*     __restrict__ block_table,
    const int*     __restrict__ bits_table,
    const int*     __restrict__ ctx_lens,
    float*         __restrict__ partial_out,
    int num_q_heads, int num_kv_heads, int kv_groups,
    int num_tiers, int max_blocks, int page_size, int dh,
    int G_max, int cb_max, int g_max,
    float sm_scale, int max_pages
){
    const int hq       = blockIdx.x;
    const int page_idx = blockIdx.y;
    if (hq >= num_q_heads) return;
    const int hkv = hq / kv_groups;
    const int tid = threadIdx.x;
    const int out_base = (hq * max_pages + page_idx) * (dh + 2);

    if (page_idx >= max_blocks) {
        if (tid == 0) { partial_out[out_base] = -INFINITY; partial_out[out_base+1] = 0.f; }
        if (tid < dh) partial_out[out_base + 2 + tid] = 0.f;
        return;
    }
    const int phys = block_table[hkv * max_blocks + page_idx];
    if (phys < 0) {
        if (tid == 0) { partial_out[out_base] = -INFINITY; partial_out[out_base+1] = 0.f; }
        if (tid < dh) partial_out[out_base + 2 + tid] = 0.f;
        return;
    }

    const int tier_idx = bits_table[hkv * max_blocks + page_idx];
    const int G_tier = tier_G_arr[tier_idx];
    const int g_tier = tier_g_arr[tier_idx];
    const int cb_stride_t = G_max * cb_max * g_max;
    const int cb_stride_g = cb_max * g_max;
    const int cb_stride_c = g_max;

    extern __shared__ char smem_raw[];
    float*   sq       = (float*)smem_raw;
    float*   s_rscale = sq + dh;
    float*   scb      = s_rscale + G_max;
    float*   s_logits = scb + G_max * cb_max * g_max;
    uint8_t* s_theta  = (uint8_t*)(s_logits + page_size);
    uint8_t* s_radius = s_theta + page_size * G_max;

    for (int i = tid; i < dh; i += PAGE_THREADS)
        sq[i] = q_all[hq * dh + i];
    if (tid < G_max)
        s_rscale[tid] = r_scales[phys * G_max + tid];
    const float* cb_src = cb_flat + hkv * (num_tiers * cb_stride_t) + tier_idx * cb_stride_t;
    const int cb_total = G_max * cb_max * g_max;
    for (int i = tid; i < cb_total; i += PAGE_THREADS)
        scb[i] = cb_src[i];
    const int code_total = page_size * G_max;
    const int theta_base = phys * page_size * G_max;
    for (int i = tid; i < code_total; i += PAGE_THREADS) {
        s_theta[i]  = theta_codes[theta_base + i];
        s_radius[i] = radius_codes[theta_base + i];
    }
    __syncthreads();

    float logit_sum = 0.f;
    if (tid < page_size) {
        for (int g = 0; g < G_tier; g++) {
            const int code = (int)s_theta[tid * G_max + g];
            const float r_c = (float)s_radius[tid * G_max + g];
            const float r_hat = s_rscale[g] * r_c / 255.0f;
            const float* q_g  = sq + g * g_tier;
            const float* cb_g = scb + g * cb_stride_g + code * cb_stride_c;
            float dot = 0.f;
            for (int d = 0; d < g_tier; d++)
                dot += q_g[d] * cb_g[d];
            logit_sum += r_hat * dot;
        }
    }
    s_logits[tid] = (logit_sum == 0.f) ? -1e9f : logit_sum * sm_scale;
    __syncthreads();

    float m_i = -INFINITY, l_i = 0.f, acc = 0.f;
    const int v_page_base = phys * page_size * dh;
    for (int t = 0; t < page_size; t++) {
        const float lg = s_logits[t];
        const float m_new = fmaxf(m_i, lg);
        const float alpha = __expf(m_i - m_new);
        const float p     = __expf(lg - m_new);
        if (tid < dh) {
            acc = alpha * acc + p * __half2float(v_pool[v_page_base + t * dh + tid]);
        }
        l_i = alpha * l_i + p;
        m_i = m_new;
    }

    if (tid == 0) {
        partial_out[out_base]     = m_i;
        partial_out[out_base + 1] = l_i;
    }
    if (tid < dh)
        partial_out[out_base + 2 + tid] = acc;
}


__global__
void sphkv_reduce_kernel(
    const float* __restrict__ partial_out,
    float*       __restrict__ out,
    int num_q_heads, int dh, int max_pages
){
    const int hq = blockIdx.x;
    if (hq >= num_q_heads) return;
    const int d = threadIdx.x;
    float m_f = -INFINITY, l_f = 0.f, a_f = 0.f;
    for (int p = 0; p < max_pages; p++) {
        const int base = (hq * max_pages + p) * (dh + 2);
        const float m_s = partial_out[base];
        const float l_s = partial_out[base + 1];
        if (m_s == -INFINITY && l_s == 0.f) continue;
        const float a_s = (d < dh) ? partial_out[base + 2 + d] : 0.f;
        const float m_new = fmaxf(m_f, m_s);
        const float alpha = __expf(m_f - m_new);
        const float beta  = __expf(m_s - m_new);
        a_f = alpha * a_f + beta * a_s;
        l_f = alpha * l_f + beta * l_s;
        m_f = m_new;
    }
    if (d < dh)
        out[hq * dh + d] = a_f / (l_f + 1e-8f);
}


void sphkv_logit_launcher(
    torch::Tensor q_all, torch::Tensor cb_flat,
    torch::Tensor tier_G_arr, torch::Tensor tier_g_arr,
    torch::Tensor theta_codes, torch::Tensor radius_codes,
    torch::Tensor r_scales, torch::Tensor v_pool,
    torch::Tensor block_table, torch::Tensor bits_table,
    torch::Tensor ctx_lens, torch::Tensor out,
    torch::Tensor partial_scratch,
    int num_q_heads, int num_kv_heads, int kv_groups,
    int num_tiers, int max_blocks, int page_size, int dh,
    int G_max, int cb_max, int g_max,
    float sm_scale, int max_ctx
){
    const int max_pages = max_blocks;
    const int smem_bytes = (dh + G_max + G_max * cb_max * g_max + page_size) * sizeof(float)
                         + page_size * G_max * 2;
    {
        dim3 grid(num_q_heads, max_pages);
        dim3 block(page_size);
        cudaFuncSetAttribute(sphkv_fused_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        sphkv_fused_kernel<<<grid, block, smem_bytes>>>(
            q_all.data_ptr<float>(), cb_flat.data_ptr<float>(),
            tier_G_arr.data_ptr<int>(), tier_g_arr.data_ptr<int>(),
            theta_codes.data_ptr<uint8_t>(), radius_codes.data_ptr<uint8_t>(),
            r_scales.data_ptr<float>(),
            (const half*)v_pool.data_ptr(),
            block_table.data_ptr<int>(), bits_table.data_ptr<int>(),
            ctx_lens.data_ptr<int>(), partial_scratch.data_ptr<float>(),
            num_q_heads, num_kv_heads, kv_groups,
            num_tiers, max_blocks, page_size, dh,
            G_max, cb_max, g_max, sm_scale, max_pages);
    }
    {
        dim3 grid(num_q_heads);
        dim3 block(PAGE_THREADS);
        sphkv_reduce_kernel<<<grid, block>>>(
            partial_scratch.data_ptr<float>(), out.data_ptr<float>(),
            num_q_heads, dh, max_pages);
    }
}
