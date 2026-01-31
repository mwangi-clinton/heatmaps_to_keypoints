#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include "decoding.h"

// Example function to mimic getting a raw pointer from a tensor object
// (In LibTorch this would be tensor.data_ptr<float>(), in TensorFlow tensor.flat<float>().data())
const float* GetTensorData(const std::vector<float>& tensor_storage) {
    return tensor_storage.data();
}

int main() {
    // 1. Setup Data: Simulate [8, 17, 96, 72] Tensor
    int BATCH = 8;
    int CHANNELS = 17;
    int HEIGHT = 96;
    int WIDTH = 72;
    
    // Total elements
    size_t total_elements = BATCH * CHANNELS * HEIGHT * WIDTH;
    std::vector<float> heatmap_tensor(total_elements, 0.0f);

    std::cout << "--- NCHW Tensor Decoding Demo ---\n";
    std::cout << "Tensor Shape: [" << BATCH << ", " << CHANNELS << ", " << HEIGHT << ", " << WIDTH << "]\n";

    // 2. Simulate a Valid Peak
    // Let's create a peak for Batch 0, Keypoint 0 (Nose) at (20.3, 30.7)
    // And perhaps another peak for Batch 1, Keypoint 0 at (40.5, 50.5)
    
    auto set_peak = [&](int b, int k, float tx, float ty) {
        // Calculate offset for this batch and keypoint
        // NCHW Layout: [Batch, Channel, Height, Width]
        size_t batch_stride  = CHANNELS * HEIGHT * WIDTH;
        size_t channel_stride = HEIGHT * WIDTH;
        
        size_t offset = (b * batch_stride) + (k * channel_stride);
        
        for (int y = 0; y < HEIGHT; ++y) {
            for (int x = 0; x < WIDTH; ++x) {
                float val = 10.0f - 1.0f * (x - tx) * (x - tx) - 1.0f * (y - ty) * (y - ty);
                heatmap_tensor[offset + y * WIDTH + x] = std::max(0.0f, val);
            }
        }
    };

    // Set peak for Batch 0, Keypoint 0
    set_peak(0, 0, 20.3f, 30.7f);
    
    // Set peak for Batch 2, Keypoint 5
    set_peak(2, 5, 45.2f, 60.8f);

    // 3. Decode Specific Batch Items
    // All you need is the content pointer and the dimensions!
    const float* raw_ptr = GetTensorData(heatmap_tensor);

    // --- Decode Batch Index 0 ---
    std::cout << "\nDecoding Batch Index 0...\n";
    std::vector<Landmark> results_b0 = DecodeHeatmaps(raw_ptr, CHANNELS, HEIGHT, WIDTH, /*batch_offset=*/ 0);
    
    // Check Keypoint 0
    Landmark& l0 = results_b0[0];
    std::cout << "  Keypoint 0 (Norm): (" << l0.x << ", " << l0.y << ") Score: " << l0.score << "\n";
    std::cout << "  Keypoint 0 (Px):   (" << l0.x * WIDTH << ", " << l0.y * HEIGHT << ")\n";

    // --- Decode Batch Index 2 ---
    // We calculate the offset: BatchIndex * (C * H * W)
    int batch_stride_elements = CHANNELS * HEIGHT * WIDTH;
    int offset_b2 = 2 * batch_stride_elements;

    std::cout << "\nDecoding Batch Index 2...\n";
    std::vector<Landmark> results_b2 = DecodeHeatmaps(raw_ptr, CHANNELS, HEIGHT, WIDTH, offset_b2);
    
    // Check Keypoint 5
    Landmark& l5 = results_b2[5];
    std::cout << "  Keypoint 5 (Norm): (" << l5.x << ", " << l5.y << ") Score: " << l5.score << "\n";
    std::cout << "  Keypoint 5 (Px):   (" << l5.x * WIDTH << ", " << l5.y * HEIGHT << ")\n";

    return 0;
}
