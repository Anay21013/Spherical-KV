#include <torch/extension.h>

void sphkv_logit_launcher(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor,
    int, int, int, int, int, int, int, int, int, int, float, int);

void sphkv_encode_append_launcher(
    torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor,
    int, int, int, int, int, int, int, int);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &sphkv_logit_launcher, "SphericalKV fused decode");
    m.def("encode_append", &sphkv_encode_append_launcher, "SphericalKV fused encode+append");
}
