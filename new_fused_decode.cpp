#include <torch/extension.h>

void sphkv_logit_launcher(
    torch::Tensor q_lut,
    torch::Tensor theta_codes,
    torch::Tensor radius_codes,
    torch::Tensor r_scales,
    torch::Tensor block_table,
    torch::Tensor bits_table,
    torch::Tensor ctx_lens,
    torch::Tensor logits_out,
    int num_q_heads,
    int num_kv_heads,
    int kv_groups,
    int num_tiers,
    int max_blocks,
    int page_size,
    int G_max,
    int cb_size_max,
    float sm_scale,
    int max_ctx
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &sphkv_logit_launcher, "SphericalKV logit kernel");
}
