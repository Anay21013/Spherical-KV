#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>
#include <torch/extension.h>

// ---------------------------------------------------------------------------
// Bit-unpacking helper: reads `bits` bits starting at `bit_offset` from a
// byte stream.  Uses a 4-byte unaligned load for safety across byte boundaries.
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// decode_kernel
//
// Grid  : (num_pages, num_q)   — one block per (page, q-head) pair
// Block : (page_size,)         — one thread per token slot in the page
//
// Shared memory layout  [cooperative load, one pass each]:
//   smem[0 .. dh)                           : sq   — q vector for this q_idx
//   smem[dh .. dh + groups*cb_size*group_size) : scb  — full codebook slice
//
// Each thread independently computes the attention logit for one KV token:
//   logit_i = (1/sqrt(dh)) * sum_g  r_hat[i,g] * dot(sq_g, cw[i,g])
//
// NO warp reduction — every thread owns exactly one token and writes one
// scalar output.  Reductions across tokens happen in the softmax layer above.
// ---------------------------------------------------------------------------
extern "C" __global__
void decode_kernel(
    const uint8_t* __restrict__ pages,
    const int*     __restrict__ pointer_table,
    const float*   __restrict__ q,          // [num_q, dh]
    int                         num_q,
    const float*   __restrict__ codebooks,  // [groups, cb_size, group_size]
    float*                      logits,     // [num_q, num_pages, page_size]
    int dh,
    int groups,
    int group_size,
    int b_theta,
    int page_size,
    int num_pages
){
    const int page_id  = blockIdx.x;
    const int token_id = threadIdx.x;
    const int q_idx    = blockIdx.y;

    // ── bounds checks ──────────────────────────────────────────────────
    if (page_id  >= num_pages) return;
    if (q_idx    >= num_q)     return;
    if (token_id >= page_size) return;

    // ── page metadata ──────────────────────────────────────────────────
    const int* ptr        = pointer_table + page_id * 3;
    const int  page_base  = ptr[0];
    const int  theta_base = ptr[1];
    const int  radius_base= ptr[2];

    const uint8_t* page  = pages + page_base;
    const int      count = (int)page[1];   // byte 1 = token_count

    if (token_id >= count) {
        // Write zero for padding slots so downstream cat/slice stays clean
        logits[q_idx * num_pages * page_size + page_id * page_size + token_id] = 0.f;
        return;
    }

    // per-group radius scales stored in header bytes [8 .. 8+G*4)
    const float* r_scales     = (const float*)(page + 8);
    const uint8_t* theta_stream = pages + theta_base;
    const int8_t*  r_stream     = (const int8_t*)(pages + radius_base);

    // ── cooperative shared-memory load ────────────────────────────────
    // Layout: | sq [dh floats] | scb [groups * cb_size * group_size floats] |
    extern __shared__ float smem[];
    float* sq  = smem;           // q vector for this q_idx: [dh]
    float* scb = smem + dh;      // codebook slice passed in: [groups, cb_size, g]

    // q vector — unique per blockIdx.y
    for (int i = threadIdx.x; i < dh; i += blockDim.x)
        sq[i] = q[q_idx * dh + i];

    // codebook — same for every token in this page/tier
    const int cb_total = groups * (1 << b_theta) * group_size;
    for (int i = threadIdx.x; i < cb_total; i += blockDim.x)
        scb[i] = codebooks[i];

    __syncthreads();   // sq and scb are fully loaded before any thread proceeds

    // ── per-token logit computation (one thread = one token) ──────────
    // logit_i = (1/sqrt(dh)) * sum_g  r_hat[i,g] * dot(sq_g, cw[theta[i,g]])
    //
    // No warp reduction: each thread is independent.  Reductions over the
    // token dimension happen in the softmax above, not here.
    float acc = 0.f;

    for (int g = 0; g < groups; g++) {
        // Unpack b_theta-bit codebook index for group g of this token
        const int bit_offset = token_id * groups * b_theta + g * b_theta;
        const int idx        = read_bits(theta_stream, bit_offset, b_theta);

        // Codeword pointer — fully in shared memory
        const float* cw = scb + g * (1 << b_theta) * group_size
                               + idx * group_size;

        // Dot product of q sub-group with codeword (both in smem → fast)
        float dot = 0.f;
        #pragma unroll
        for (int i = 0; i < group_size; i++)
            dot += sq[g * group_size + i] * cw[i];

        // Dequantise radius and accumulate
        const int8_t r_code = r_stream[token_id * groups + g];
        const float  r_hat  = r_scales[g] * (float)r_code / 127.0f;
        acc += r_hat * dot;
    }

    // Write logit — every active thread writes exactly one scalar
    logits[q_idx * num_pages * page_size + page_id * page_size + token_id]
        = acc * rsqrtf((float)dh);
}

void decode_kernel_launcher(
    torch::Tensor pages,
    torch::Tensor pointer_table,
    torch::Tensor q,          // [num_q, dh]
    torch::Tensor codebooks,  // [groups, cb_size, group_size]
    torch::Tensor logits,     // [num_q, num_pages, page_size]
    int dh,
    int groups,
    int group_size,
    int b_theta,
    int page_size
){
    const int num_pages = pointer_table.size(0);
    const int num_q     = q.size(0);

    dim3 grid(num_pages, num_q);
    dim3 block(page_size);

    // Shared memory: q vector + full codebook slice for this tier
    const int shm_bytes = (dh + groups * (1 << b_theta) * group_size)
                          * sizeof(float);

    decode_kernel<<<grid, block, shm_bytes>>>(
        pages.data_ptr<uint8_t>(),
        pointer_table.data_ptr<int>(),
        q.data_ptr<float>(),
        num_q,
        codebooks.data_ptr<float>(),
        logits.data_ptr<float>(),
        dh,
        groups,
        group_size,
        b_theta,
        page_size,
        num_pages
    );
}