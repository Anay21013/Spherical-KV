#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>
#include <torch/extension.h>

// Maximum threads per block = tile size for the token-processing loop.
#define MAX_TILE_SZ 256

__device__ __forceinline__ int read_bits(
    const uint8_t* data,
    int            bit_offset,
    int            bits
){
    int      byte_offset = bit_offset >> 3;
    int      bit_shift   = bit_offset & 7;
    uint32_t chunk;
    memcpy(&chunk, data + byte_offset, sizeof(uint32_t));
    chunk >>= bit_shift;
    return chunk & ((1 << bits) - 1);
}

extern "C" __global__
void decode_kernel(
    const uint8_t* __restrict__ pages,
    const int*     __restrict__ pointer_table,
    const int*     __restrict__ positions,
    const float*   __restrict__ cos_table,
    const float*   __restrict__ sin_table,
    const float*   __restrict__ q,
    int                         num_q,
    const float*   __restrict__ codebooks,
    float*                      logits,
    int dh,
    int groups,
    int group_size,
    int b_theta,
    int page_size,
    int num_pages,
    int max_pos
){
    const int page_id = blockIdx.x;
    const int q_idx   = blockIdx.y;
    if (page_id >= num_pages || q_idx >= num_q) return;

    // -- page metadata ---------------------------------------------------
    const int* ptr         = pointer_table + page_id * 3;
    const int  page_base   = ptr[0];
    const int  theta_base  = ptr[1];
    const int  radius_base = ptr[2];

    const uint8_t* page  = pages + page_base;
    const int      count = (int)page[1];          // valid tokens in this page

    const float*   r_scales      = (const float*)(page + 8);
    const uint8_t* theta_stream  = pages + theta_base;
    const int8_t*  radius_stream = (const int8_t*)(pages + radius_base);

    // Per-page positions slice
    const int* page_positions = positions + page_id * page_size;

    extern __shared__ float smem[];
    float* sq  = smem;                                             // [dh]

    const int cb_size  = 1 << b_theta;
    const int cb_total = groups * cb_size * group_size;
    float* scb = smem + dh;                                        // [cb_total]

    const int tile_size         = blockDim.x;
    const int tile_theta_alloc  = (tile_size * groups * b_theta + 7) / 8 + 4;
    uint8_t* s_theta  = (uint8_t*)(scb + cb_total);
    int8_t*  s_radius = (int8_t*)(s_theta + tile_theta_alloc);

    const int out_base = q_idx * num_pages * page_size + page_id * page_size;

    for (int i = threadIdx.x; i < dh; i += blockDim.x)
        sq[i] = q[q_idx * dh + i];
    __syncthreads();

    for (int i = threadIdx.x; i < cb_total; i += blockDim.x)
        scb[i] = codebooks[i];
    __syncthreads();

    const int total_theta_bytes = (count * groups * b_theta + 7) / 8;
    const int num_tiles         = (count + tile_size - 1) / tile_size;
    const float inv_sqrt_dh     = rsqrtf((float)dh);
    const int   dh_half         = dh >> 1;

    for (int tile = 0; tile < num_tiles; tile++) {
        const int t_start = tile * tile_size;
        const int t_count = min(tile_size, count - t_start);

        const int tile_bit_start = t_start * groups * b_theta;
        const int byte_start     = tile_bit_start >> 3;
        const int byte_end_base  = ((t_start + t_count) * groups * b_theta + 7) >> 3;
        const int byte_end       = min(byte_end_base + 3, total_theta_bytes + 3);
        int       theta_load     = byte_end - byte_start;
        if (theta_load > tile_theta_alloc) theta_load = tile_theta_alloc;

        for (int i = threadIdx.x; i < theta_load; i += blockDim.x)
            s_theta[i] = theta_stream[byte_start + i];
        for (int i = theta_load + threadIdx.x; i < tile_theta_alloc; i += blockDim.x)
            s_theta[i] = 0;

        const int r_load = t_count * groups;
        const int8_t* tile_r_src = radius_stream + t_start * groups;
        for (int i = threadIdx.x; i < r_load; i += blockDim.x)
            s_radius[i] = tile_r_src[i];

        __syncthreads();

        if (threadIdx.x < t_count) {
            const int token_in_page = t_start + threadIdx.x;
            const int abs_pos       = page_positions[token_in_page];

            if (abs_pos < 0 || abs_pos >= max_pos) {
                logits[out_base + token_in_page] = 0.f;
            } else {
                const float* cos_p = cos_table + abs_pos * dh;
                const float* sin_p = sin_table + abs_pos * dh;

                float acc = 0.f;
                for (int g = 0; g < groups; g++) {
                    const int global_bit = token_in_page * groups * b_theta + g * b_theta;
                    const int local_bit  = global_bit - (byte_start << 3);
                    const int idx        = read_bits(s_theta, local_bit, b_theta);

                    const float* cw = scb + g * cb_size * group_size
                                          + idx * group_size;

                    float dot = 0.f;
                    #pragma unroll
                    for (int d = 0; d < group_size; d++) {
                        const int i_glob   = g * group_size + d;
                        const float q_i    = sq[i_glob];
                        const float q_half = (i_glob < dh_half)
                                             ? -sq[i_glob + dh_half]
                                             :  sq[i_glob - dh_half];
                        const float c_i    = cos_p[i_glob];
                        const float s_i    = sin_p[i_glob];
                        const float q_rot  = q_i * c_i - q_half * s_i;
                        dot += q_rot * cw[d];
                    }

                    // Dequantize radius and accumulate
                    const int8_t r_code = s_radius[threadIdx.x * groups + g];
                    const float  r_hat  = r_scales[g] * (float)r_code / 127.0f;
                    acc += r_hat * dot;
                }
                logits[out_base + token_in_page] = acc * inv_sqrt_dh;
            }
        }
        __syncthreads();
    }

    // Zero padding slots beyond count 
    for (int i = count + threadIdx.x; i < page_size; i += blockDim.x)
        logits[out_base + i] = 0.f;
}


void decode_kernel_launcher(
    torch::Tensor pages,
    torch::Tensor pointer_table,
    torch::Tensor positions,
    torch::Tensor cos_table,
    torch::Tensor sin_table,
    torch::Tensor q,
    torch::Tensor codebooks,
    torch::Tensor logits,
    int dh,
    int groups,
    int group_size,
    int b_theta,
    int page_size
){
    const int num_pages = pointer_table.size(0);
    const int num_q     = q.size(0);
    const int max_pos   = (int)cos_table.size(0);

    const int tile_sz = (page_size <= MAX_TILE_SZ) ? page_size : MAX_TILE_SZ;
    dim3 grid(num_pages, num_q);
    dim3 block(tile_sz);

    const int cb_total          = groups * (1 << b_theta) * group_size;
    const int float_bytes       = (dh + cb_total) * (int)sizeof(float);
    const int tile_theta_bytes  = (tile_sz * groups * b_theta + 7) / 8 + 4;
    const int tile_radius_bytes = tile_sz * groups;
    const int shm_bytes         = float_bytes + tile_theta_bytes + tile_radius_bytes;

    decode_kernel<<<grid, block, shm_bytes>>>(
        pages.data_ptr<uint8_t>(),
        pointer_table.data_ptr<int>(),
        positions.data_ptr<int>(),
        cos_table.data_ptr<float>(),
        sin_table.data_ptr<float>(),
        q.data_ptr<float>(),
        num_q,
        codebooks.data_ptr<float>(),
        logits.data_ptr<float>(),
        dh,
        groups,
        group_size,
        b_theta,
        page_size,
        num_pages,
        max_pos
    );
}