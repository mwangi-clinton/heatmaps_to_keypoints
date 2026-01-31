#include "decoding.h"

#include <algorithm>
#include <cmath>
#include <vector>
#include <iostream> 

// Standalone C++ implementation - No MediaPipe dependencies

std::vector<Landmark> DecodeHeatmaps(const float* data, int num_keypoints,
                                     int height, int width, int batch_offset) {
  std::vector<Landmark> landmarks;
  landmarks.reserve(num_keypoints);

  int channel_stride = height * width;

  for (int k = 0; k < num_keypoints; ++k) {
    // Offset to the start of this keypoint's heatmap
    const float* heatmap_start = data + batch_offset + (k * channel_stride);
    const float* heatmap_end = heatmap_start + channel_stride;

    // --- A. FIND ARGMAX (Optimized: Linear scan) ---
    // std::max_element is extremely efficient for finding the max in a contiguous block
    auto max_it = std::max_element(heatmap_start, heatmap_end);
    float max_val = *max_it;

    int max_idx = std::distance(heatmap_start, max_it);
    int py = max_idx / width;
    int px = max_idx % width;

    float final_x = static_cast<float>(px);
    float final_y = static_cast<float>(py);

    // --- B. APPLY REFINEMENT (Taylor Expansion / Hessian) ---
    // Bounds check: We need space for neighbors (at least 2 pixels from edge)
    // This is the "Unbiased" method from the original code
    if (px > 1 && px < width - 2 && py > 1 && py < height - 2) {
      // Helper lambda to get value at relative (x,y)
      auto get_val = [&](int x, int y) -> float {
        return heatmap_start[y * width + x];
      };

      float v_c = max_val;
      // 1st Derivative
      float dx = 0.5f * (get_val(px + 1, py) - get_val(px - 1, py));
      float dy = 0.5f * (get_val(px, py + 1) - get_val(px, py - 1));

      // 2nd Derivative (Hessian)
      float dxx = 0.25f * (get_val(px + 2, py) - 2 * v_c + get_val(px - 2, py));
      float dyy = 0.25f * (get_val(px, py + 2) - 2 * v_c + get_val(px, py - 2));

      // Cross derivative (dxy)
      float dxy = 0.25f * (get_val(px + 1, py + 1) - get_val(px + 1, py - 1) -
                           get_val(px - 1, py + 1) + get_val(px - 1, py - 1));

      // Invert Hessian (2x2 Matrix Inversion)
      float det = (dxx * dyy) - (dxy * dxy);

      if (std::abs(det) > 1e-6) {
        float inv_dxx = dyy / det;
        float inv_dyy = dxx / det;
        float inv_dxy = -dxy / det;

        // Offset = -HessianInv * Derivative
        float offset_x = -(inv_dxx * dx + inv_dxy * dy);
        float offset_y = -(inv_dxy * dx + inv_dyy * dy);

        final_x += offset_x;
        final_y += offset_y;
      }
    }

    // Return NORMALIZED coordinates (0.0 to 1.0)
    landmarks.push_back({final_x / static_cast<float>(width),
                         final_y / static_cast<float>(height), max_val});
  }

  return landmarks;
}
