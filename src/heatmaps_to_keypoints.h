#pragma once

#include <vector>
#include <array>
#include <string>
#include <opencv2/opencv.hpp>

using namespace cv;

using Box = std::array<float, 4>;  // [x1, y1, x2, y2]
using Vec2 = std::array<float, 2>; // [x, y]
using Centers = std::vector<Vec2>;
using Scales = std::vector<Vec2>;

struct CentersAndScales
{
    Centers centers;
    Scales scales;
};

using Heatmaps3D = std::vector<Mat>;        // K x (H rows x W cols, CV_32F)
using Heatmaps4D = std::vector<Heatmaps3D>; // N x K x (H x W)
using Coords2D = std::vector<Vec2>;         // K x 2
using Coords3D = std::vector<Coords2D>;     // N x K x 2
using Maxvals2D = std::vector<float>;       // K x 1 (as float)
using Maxvals3D = std::vector<Maxvals2D>;   // N x K x 1

struct KeypointsResult
{
    Coords3D preds;
    Maxvals3D maxvals;
};

// Function declarations
CentersAndScales get_centers_and_scales_xyxy(const std::vector<Box> &person_boxes, float scale_factor = 1.0f);

KeypointsResult keypoints_from_heatmaps(
    Heatmaps4D heatmaps,
    const Centers &center,
    const Scales &scale,
    bool unbiased = false,
    std::string post_process = "default",
    int kernel = 11,
    float valid_radius_factor = 0.0546875f,
    bool use_udp = false,
    std::string target_type = "GaussianHeatmap");

// Wrapper
KeypointsResult keypoints_from_heatmaps_with_boxes(
    Heatmaps4D heatmaps,
    const std::vector<Box> &boxes,
    float scale_factor = 1.0f,
    bool unbiased = false,
    std::string post_process = "default",
    int kernel = 11,
    float valid_radius_factor = 0.0546875f,
    bool use_udp = false,
    std::string target_type = "GaussianHeatmap");
