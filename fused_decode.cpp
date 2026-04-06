#include <torch/extension.h>

void decode_kernel_launcher(
    torch::Tensor pages,
    torch::Tensor pointer_table,
    torch::Tensor q,
    torch::Tensor codebooks,
    torch::Tensor logits,
    int dh,
    int groups,
    int group_size,
    int b_theta,
    int page_size
);

void fused_decode_forward(
    torch::Tensor pages,
    torch::Tensor pointer_table,
    torch::Tensor q,
    torch::Tensor codebooks,
    torch::Tensor logits,
    int dh,
    int groups,
    int group_size,
    int b_theta,
    int page_size
){
    decode_kernel_launcher(
        pages, pointer_table, q, codebooks, logits,
        dh, groups, group_size, b_theta, page_size
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fused_decode_forward, "Fused decode kernel (batched q)");
}