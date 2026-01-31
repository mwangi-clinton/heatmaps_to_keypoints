#ifndef DECODING_H_
#define DECODING_H_

#include <vector>

struct Landmark {
  float x;
  float y;
  float score;
};

// Decodes heatmaps into landmarks using Sub-pixel Taylor refinement.
// Expected Input: NCHW memory layout (contiguous keypoints).
// batch_offset: Offset in the float array to start reading from (e.g. for batch index > 0).
std::vector<Landmark> DecodeHeatmaps(const float* data, int num_keypoints,
                                     int height, int width, int batch_offset = 0);

#endif  // DECODING_H_
