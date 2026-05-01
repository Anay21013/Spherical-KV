#include <torch/extension.h>
void sphkv_logit_launcher(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    int, int, int, int, int, int, int, int, int, int, float, int);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &sphkv_logit_launcher, "SphericalKV logit kernel (inline dot)");
}
