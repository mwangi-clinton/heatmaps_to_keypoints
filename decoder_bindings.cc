#include <torch/extension.h>
#include "decoding.h"

std::vector<std::vector<Landmark>> decode_heatmaps_torch(
    torch::Tensor heatmaps
) {
  // Ensure tensor is correct
  heatmaps = heatmaps.contiguous();
  TORCH_CHECK(heatmaps.device().is_cpu(), "CPU tensor required");
  TORCH_CHECK(heatmaps.dtype() == torch::kFloat32, "float32 required");

  int batch = heatmaps.size(0);
  int num_keypoints = heatmaps.size(1);
  int height = heatmaps.size(2);
  int width = heatmaps.size(3);

  const float* data = heatmaps.data_ptr<float>();

  int batch_stride = num_keypoints * height * width;

  std::vector<std::vector<Landmark>> all_results;
  all_results.reserve(batch);

  for (int b = 0; b < batch; ++b) {
    int batch_offset = b * batch_stride;
    all_results.push_back(
      DecodeHeatmaps(data, num_keypoints, height, width, batch_offset)
    );
  }

  return all_results;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("decode_heatmaps", &decode_heatmaps_torch,
        "Decode heatmaps (CPU)");
}
