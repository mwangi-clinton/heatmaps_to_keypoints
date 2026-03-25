/**
 * keypoint_heatmap.cpp
 *
 * C++ port of the Python heatmap-based keypoint decoding utilities.
 * Dependencies: OpenCV (for GaussianBlur)
 */

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

struct Array4D {
    std::vector<float> data;
    int N = 0, K = 0, H = 0, W = 0;

    Array4D() = default;
    Array4D(int n, int k, int h, int w, float val = 0.f)
        : N(n), K(k), H(h), W(w),
          data(static_cast<size_t>(n) * k * h * w, val) {}

    float& at(int n, int k, int h, int w) {
        return data[static_cast<size_t>(n) * K * H * W
                  + static_cast<size_t>(k) * H * W
                  + static_cast<size_t>(h) * W + w];
    }
    float at(int n, int k, int h, int w) const {
        return data[static_cast<size_t>(n) * K * H * W
                  + static_cast<size_t>(k) * H * W
                  + static_cast<size_t>(h) * W + w];
    }
};

struct Array3D {
    std::vector<float> data;
    int N = 0, K = 0, D = 0;

    Array3D() = default;
    Array3D(int n, int k, int d, float val = 0.f)
        : N(n), K(k), D(d),
          data(static_cast<size_t>(n) * k * d, val) {}

    float& at(int n, int k, int d) {
        return data[static_cast<size_t>(n) * K * D
                  + static_cast<size_t>(k) * D + d];
    }
    float at(int n, int k, int d) const {
        return data[static_cast<size_t>(n) * K * D
                  + static_cast<size_t>(k) * D + d];
    }
};

// ============================================================
// gaussian_blur
// ============================================================

Array4D gaussian_blur(Array4D heatmaps, int kernel = 11) {
    assert(kernel % 2 == 1 && "kernel must be odd");
    const int border = (kernel - 1) / 2;
    const int N = heatmaps.N, K = heatmaps.K;
    const int H = heatmaps.H, W = heatmaps.W;

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < K; ++j) {
            float origin_max = -std::numeric_limits<float>::infinity();
            for (int h = 0; h < H; ++h)
                for (int w = 0; w < W; ++w)
                    origin_max = std::max(origin_max, heatmaps.at(i, j, h, w));

            // FIXED: Removed cv:: before CV_32F
            cv::Mat dr(H + 2 * border, W + 2 * border, CV_32F, cv::Scalar(0.f));
            for (int h = 0; h < H; ++h)
                for (int w = 0; w < W; ++w)
                    dr.at<float>(h + border, w + border) = heatmaps.at(i, j, h, w);

            cv::GaussianBlur(dr, dr, cv::Size(kernel, kernel), 0);

            float new_max = -std::numeric_limits<float>::infinity();
            for (int h = 0; h < H; ++h)
                for (int w = 0; w < W; ++w)
                    new_max = std::max(new_max, dr.at<float>(h + border, w + border));

            const float scale_fac = (new_max > 0.f) ? (origin_max / new_max) : 1.f;

            for (int h = 0; h < H; ++h)
                for (int w = 0; w < W; ++w)
                    heatmaps.at(i, j, h, w) =
                        dr.at<float>(h + border, w + border) * scale_fac;
        }
    }
    return heatmaps;
}

// ============================================================
// get_max_preds
// ============================================================

std::pair<Array3D, Array3D> get_max_preds(const Array4D& heatmaps) {
    const int N = heatmaps.N, K = heatmaps.K;
    const int H = heatmaps.H, W = heatmaps.W;

    Array3D preds(N, K, 2, 0.f);
    Array3D maxvals(N, K, 1, 0.f);

    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            float max_val = -std::numeric_limits<float>::infinity();
            int   max_idx = 0;
            const size_t base =
                static_cast<size_t>(n) * K * H * W +
                static_cast<size_t>(k) * H * W;

            for (int hw = 0; hw < H * W; ++hw) {
                const float v = heatmaps.data[base + hw];
                if (v > max_val) { max_val = v; max_idx = hw; }
            }

            maxvals.at(n, k, 0) = max_val;
            if (max_val > 0.f) {
                preds.at(n, k, 0) = static_cast<float>(max_idx % W);
                preds.at(n, k, 1) = static_cast<float>(max_idx / W);
            } else {
                preds.at(n, k, 0) = -1.f;
                preds.at(n, k, 1) = -1.f;
            }
        }
    }
    return {preds, maxvals};
}

// ============================================================
// transform_preds
// ============================================================

void transform_preds(
    std::vector<float>& coords,
    int rows, int ndims,
    const float center[2],
    const float scale[2],
    const int   output_size[2],
    bool use_udp = false
) {
    const float sx = scale[0] * 200.f;
    const float sy = scale[1] * 200.f;

    const float scale_x = use_udp
        ? sx / (output_size[0] - 1.f)
        : sx / static_cast<float>(output_size[0]);
    const float scale_y = use_udp
        ? sy / (output_size[1] - 1.f)
        : sy / static_cast<float>(output_size[1]);

    for (int r = 0; r < rows; ++r) {
        coords[r * ndims + 0] =
            coords[r * ndims + 0] * scale_x + center[0] - sx * 0.5f;
        coords[r * ndims + 1] =
            coords[r * ndims + 1] * scale_y + center[1] - sy * 0.5f;
    }
}

// ============================================================
// post_dark_udp
// ============================================================

Array3D post_dark_udp(Array3D coords, Array4D batch_heatmaps, int kernel = 3) {
    const int B = batch_heatmaps.N, K = batch_heatmaps.K;
    const int H = batch_heatmaps.H, W = batch_heatmaps.W;
    const int N = coords.N;
    assert((B == 1 || B == N) && "B must be 1 or equal to N");

    for (int b = 0; b < B; ++b) {
        for (int k = 0; k < K; ++k) {
            // FIXED: Removed cv:: before CV_32F
            cv::Mat hm(H, W, CV_32F);
            for (int h = 0; h < H; ++h)
                for (int w = 0; w < W; ++w)
                    hm.at<float>(h, w) = batch_heatmaps.at(b, k, h, w);
            cv::GaussianBlur(hm, hm, cv::Size(kernel, kernel), 0);
            for (int h = 0; h < H; ++h)
                for (int w = 0; w < W; ++w)
                    batch_heatmaps.at(b, k, h, w) = hm.at<float>(h, w);
        }
    }

    for (auto& v : batch_heatmaps.data)
        v = std::log(std::max(std::min(v, 50.f), 0.001f));

    const int pH = H + 2, pW = W + 2;
    std::vector<float> padded(static_cast<size_t>(B) * K * pH * pW);

    for (int b = 0; b < B; ++b) {
        for (int k = 0; k < K; ++k) {
            for (int h = 0; h < pH; ++h) {
                const int sh = std::max(0, std::min(H - 1, h - 1));
                for (int w = 0; w < pW; ++w) {
                    const int sw = std::max(0, std::min(W - 1, w - 1));
                    padded[static_cast<size_t>(b) * K * pH * pW
                         + static_cast<size_t>(k) * pH * pW
                         + h * pW + w] =
                        batch_heatmaps.at(b, k, sh, sw);
                }
            }
        }
    }

    const float eps = std::numeric_limits<float>::epsilon();

    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            const int b = (B == 1) ? 0 : n;
            int cx = static_cast<int>(coords.at(n, k, 0)) + 1;
            int cy = static_cast<int>(coords.at(n, k, 1)) + 1;
            cx = std::max(1, std::min(pW - 2, cx));
            cy = std::max(1, std::min(pH - 2, cy));

            const size_t base =
                static_cast<size_t>(b) * K * pH * pW +
                static_cast<size_t>(k) * pH * pW;
            const size_t id = base + static_cast<size_t>(cy) * pW + cx;

            const float i_      = padded[id];
            const float ix1     = padded[id + 1];
            const float iy1     = padded[id + pW];
            const float ix1y1   = padded[id + pW + 1];
            const float ix1_y1_ = padded[id - pW - 1];
            const float ix1_    = padded[id - 1];
            const float iy1_    = padded[id - pW]; 

            const float dx = 0.5f * (ix1  - ix1_);
            const float dy = 0.5f * (iy1  - iy1_);

            const float dxx = ix1  - 2.f * i_ + ix1_;
            const float dyy = iy1  - 2.f * i_ + iy1_;
            const float dxy = 0.5f * (ix1y1 - ix1 - iy1 + i_
                                    + i_    - ix1_ - iy1_ + ix1_y1_);

            const float a = dxx + eps;
            const float b_v = dxy; 
            const float d_v = dyy + eps;

            float det = a * d_v - b_v * b_v;
            if (std::abs(det) < 1e-15f)
                det = (det >= 0.f) ? 1e-15f : -1e-15f;

            const float ia  =  d_v / det;
            const float ib  = -b_v / det;
            const float id2 =  a   / det;

            coords.at(n, k, 0) -= ia  * dx + ib  * dy;
            coords.at(n, k, 1) -= ib  * dx + id2 * dy; 
        }
    }
    return coords;
}

// ============================================================
// taylor_inplace
// ============================================================

void taylor_inplace(
    const std::vector<float>& heatmap, 
    int H, int W,
    float& px_f, float& py_f
) {
    const int px = static_cast<int>(px_f);
    const int py = static_cast<int>(py_f);

    if (px > 1 && px < W - 2 && py > 1 && py < H - 2) {
        auto hm = [&](int y, int x) {
            return heatmap[static_cast<size_t>(y) * W + x];
        };

        const float dx  = 0.5f  * (hm(py, px + 1) - hm(py, px - 1));
        const float dy  = 0.5f  * (hm(py + 1, px) - hm(py - 1, px));
        const float dxx = 0.25f * (hm(py, px + 2) - 2.f * hm(py, px)
                                 + hm(py, px - 2));
        const float dxy = 0.25f * ( hm(py + 1, px + 1)
                                  - hm(py - 1, px + 1)
                                  - hm(py + 1, px - 1)
                                  + hm(py - 1, px - 1));
        const float dyy = 0.25f * (hm(py + 2, px) - 2.f * hm(py, px)
                                 + hm(py - 2, px));

        const float det = dxx * dyy - dxy * dxy;
        if (det != 0.f) {
            px_f += -( dyy * dx - dxy * dy) / det;
            py_f += -(-dxy * dx + dxx * dy) / det;
        }
    }
}

// ============================================================
// keypoints_from_heatmaps
// ============================================================

std::pair<Array3D, Array3D> keypoints_from_heatmaps(
    Array4D heatmaps,
    const std::vector<std::array<float, 2>>& center,
    const std::vector<std::array<float, 2>>& scale,
    bool        unbiased            = false,
    std::string post_process        = "default",
    int         kernel              = 11,
    float       valid_radius_factor = 0.0546875f,
    bool        use_udp             = false,
    std::string target_type         = "GaussianHeatmap"
) {
    if (unbiased) {
        assert(post_process != "megvii"
            && post_process != "null"
            && !post_process.empty());
    }
    if (post_process == "megvii" || post_process == "unbiased") {
        assert(kernel > 0);
    }
    if (use_udp) {
        assert(post_process != "megvii");
    }

    if (post_process == "default" && unbiased) {
        post_process = "unbiased";
    }

    if (post_process == "megvii") {
        heatmaps = gaussian_blur(heatmaps, kernel);
    }

    int N = heatmaps.N, K = heatmaps.K;
    const int H = heatmaps.H, W = heatmaps.W;

    Array3D preds, maxvals;

    std::string tt = target_type;
    std::transform(tt.begin(), tt.end(), tt.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    if (use_udp) {
        if (tt == "gaussianheatmap") {
            auto [p, m] = get_max_preds(heatmaps);
            preds = p; maxvals = m;
            preds = post_dark_udp(preds, heatmaps, kernel);

        } else if (tt == "combinedtarget") {
            for (int n = 0; n < N; ++n) {
                for (int i = 0; i < K; ++i) {
                    const int kt = (i % 3 == 0) ? (2 * kernel + 1) : kernel;
                    // FIXED: Removed cv:: before CV_32F
                    cv::Mat hm(H, W, CV_32F);
                    for (int h = 0; h < H; ++h)
                        for (int w = 0; w < W; ++w)
                            hm.at<float>(h, w) = heatmaps.at(n, i, h, w);
                    cv::GaussianBlur(hm, hm, cv::Size(kt, kt), 0);
                    for (int h = 0; h < H; ++h)
                        for (int w = 0; w < W; ++w)
                            heatmaps.at(n, i, h, w) = hm.at<float>(h, w);
                }
            }

            const float valid_radius = valid_radius_factor * static_cast<float>(H);

            std::vector<float> offset_x, offset_y;
            for (int n = 0; n < N; ++n)
                for (int k = 1; k < K; k += 3)
                    for (int h = 0; h < H; ++h)
                        for (int w = 0; w < W; ++w)
                            offset_x.push_back(heatmaps.at(n, k, h, w) * valid_radius);

            for (int n = 0; n < N; ++n)
                for (int k = 2; k < K; k += 3)
                    for (int h = 0; h < H; ++h)
                        for (int w = 0; w < W; ++w)
                            offset_y.push_back(heatmaps.at(n, k, h, w) * valid_radius);

            const int newK = K / 3;
            Array4D hm_reduced(N, newK, H, W);
            for (int n = 0; n < N; ++n)
                for (int ki = 0; ki < newK; ++ki)
                    for (int h = 0; h < H; ++h)
                        for (int w = 0; w < W; ++w)
                            hm_reduced.at(n, ki, h, w) =
                                heatmaps.at(n, ki * 3, h, w);

            auto [p, m] = get_max_preds(hm_reduced);
            preds = p; maxvals = m;
            K = newK;   

            for (int n = 0; n < N; ++n) {
                for (int ki = 0; ki < newK; ++ki) {
                    const int flat =
                        static_cast<int>(preds.at(n, ki, 0))
                      + static_cast<int>(preds.at(n, ki, 1)) * W
                      + W * H * (n * newK + ki);

                    if (flat >= 0 && flat < static_cast<int>(offset_x.size()))
                        preds.at(n, ki, 0) += offset_x[flat];
                    if (flat >= 0 && flat < static_cast<int>(offset_y.size()))
                        preds.at(n, ki, 1) += offset_y[flat];
                }
            }

        } else {
            throw std::invalid_argument(
                "target_type must be 'GaussianHeatmap' or 'CombinedTarget'");
        }

    } else {
        auto [p, m] = get_max_preds(heatmaps);
        preds = p; maxvals = m;

        if (post_process == "unbiased") {
            heatmaps = gaussian_blur(heatmaps, kernel);
            for (auto& v : heatmaps.data)
                v = std::log(std::max(v, 1e-10f));

            for (int n = 0; n < N; ++n) {
                for (int k = 0; k < K; ++k) {
                    std::vector<float> hm(static_cast<size_t>(H) * W);
                    for (int h = 0; h < H; ++h)
                        for (int w = 0; w < W; ++w)
                            hm[static_cast<size_t>(h) * W + w] =
                                heatmaps.at(n, k, h, w);
                    taylor_inplace(hm, H, W,
                                   preds.at(n, k, 0),
                                   preds.at(n, k, 1));
                }
            }

        } else if (!post_process.empty() && post_process != "null") {
            for (int n = 0; n < N; ++n) {
                for (int k = 0; k < K; ++k) {
                    const int px = static_cast<int>(preds.at(n, k, 0));
                    const int py = static_cast<int>(preds.at(n, k, 1));
                    if (px > 1 && px < W - 1 && py > 1 && py < H - 1) {
                        const float diff_x =
                            heatmaps.at(n, k, py, px + 1) -
                            heatmaps.at(n, k, py, px - 1);
                        const float diff_y =
                            heatmaps.at(n, k, py + 1, px) -
                            heatmaps.at(n, k, py - 1, px);
                        preds.at(n, k, 0) +=
                            (diff_x > 0.f) ? 0.25f : (diff_x < 0.f ? -0.25f : 0.f);
                        preds.at(n, k, 1) +=
                            (diff_y > 0.f) ? 0.25f : (diff_y < 0.f ? -0.25f : 0.f);
                        if (post_process == "megvii") {
                            preds.at(n, k, 0) += 0.5f;
                            preds.at(n, k, 1) += 0.5f;
                        }
                    }
                }
            }
        }
    }

    const int finalN = preds.N;
    const int finalK = preds.K;
    const int output_size[2] = {W, H};

    for (int i = 0; i < finalN; ++i) {
        std::vector<float> coords_i(static_cast<size_t>(finalK) * 2);
        for (int k = 0; k < finalK; ++k) {
            coords_i[k * 2 + 0] = preds.at(i, k, 0);
            coords_i[k * 2 + 1] = preds.at(i, k, 1);
        }

        transform_preds(coords_i, finalK, 2,
                        center[i].data(), scale[i].data(),
                        output_size, use_udp);

        for (int k = 0; k < finalK; ++k) {
            preds.at(i, k, 0) = coords_i[k * 2 + 0];
            preds.at(i, k, 1) = coords_i[k * 2 + 1];
        }
    }

    if (post_process == "megvii") {
        for (auto& v : maxvals.data)
            v = v / 255.f + 0.5f;
    }

    return {preds, maxvals};
}

#include <torch/extension.h>

// Helper function to convert torch::Tensor to Array4D
Array4D tensor_to_array4d(torch::Tensor t) {
    t = t.contiguous();
    TORCH_CHECK(t.dim() == 4, "heatmaps must be 4D");
    TORCH_CHECK(t.scalar_type() == torch::kFloat32, "heatmaps must be float32");
    int N = t.size(0);
    int K = t.size(1);
    int H = t.size(2);
    int W = t.size(3);
    Array4D arr;
    arr.N = N; arr.K = K; arr.H = H; arr.W = W;
    arr.data.assign(t.data_ptr<float>(), t.data_ptr<float>() + N * K * H * W);
    return arr;
}

// Helper function to convert Array3D to torch::Tensor
torch::Tensor array3d_to_tensor(const Array3D& arr) {
    auto t = torch::empty({arr.N, arr.K, arr.D}, torch::kFloat32);
    std::memcpy(t.data_ptr<float>(), arr.data.data(), arr.data.size() * sizeof(float));
    return t;
}

std::vector<std::array<float, 2>> tensor_to_vec_array(torch::Tensor t) {
    t = t.contiguous();
    TORCH_CHECK(t.dim() == 2 && t.size(1) == 2, "Tensor must be Nx2");
    TORCH_CHECK(t.scalar_type() == torch::kFloat32, "Tensor must be float32");
    int N = t.size(0);
    std::vector<std::array<float, 2>> res(N);
    const float* ptr = t.data_ptr<float>();
    for (int i = 0; i < N; ++i) {
        res[i][0] = ptr[i * 2 + 0];
        res[i][1] = ptr[i * 2 + 1];
    }
    return res;
}

// Wrapper for PyBind11
std::tuple<torch::Tensor, torch::Tensor> keypoints_from_heatmaps_wrapper(
    torch::Tensor heatmaps,
    torch::Tensor center,
    torch::Tensor scale,
    bool unbiased = false,
    std::string post_process = "default",
    int kernel = 11,
    float valid_radius_factor = 0.0546875f,
    bool use_udp = false,
    std::string target_type = "GaussianHeatmap"
) {
    Array4D hm_arr = tensor_to_array4d(heatmaps);
    auto center_vec = tensor_to_vec_array(center);
    auto scale_vec = tensor_to_vec_array(scale);

    auto result = keypoints_from_heatmaps(
        hm_arr, center_vec, scale_vec, 
        unbiased, post_process, kernel, valid_radius_factor, use_udp, target_type
    );

    return std::make_tuple(array3d_to_tensor(result.first), array3d_to_tensor(result.second));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "HeatmapToKeypoints: A high-performance C++/PyTorch extension for decoding 2D heatmaps into precise keypoint coordinates.";

    m.def("keypoints_from_heatmaps", &keypoints_from_heatmaps_wrapper,
          R"pbdoc(
Decode heatmaps to keypoints using sub-pixel refinement.

Args:
    heatmaps (torch.Tensor): A 4D float32 tensor of shape [Batch, Keypoints, Height, Width].
    center (torch.Tensor): A 2D float32 tensor of shape [Batch, 2] containing center coordinates.
    scale (torch.Tensor): A 2D float32 tensor of shape [Batch, 2] containing scale values.
    unbiased (bool, optional): Whether to use unbiased decoding. Defaults to False.
    post_process (str, optional): Post-processing method ("default", "unbiased", "megvii"). Defaults to "default".
    kernel (int, optional): Gaussian blur kernel size. Must be odd. Defaults to 11.
    valid_radius_factor (float, optional): Valid radius factor for UDP. Defaults to 0.0546875.
    use_udp (bool, optional): Whether to use UDP (Unbiased Data Processing). Defaults to False.
    target_type (str, optional): Target type ("GaussianHeatmap", "CombinedTarget"). Defaults to "GaussianHeatmap".

Returns:
    tuple[torch.Tensor, torch.Tensor]: A tuple containing:
        - preds: A 3D tensor of shape [Batch, Keypoints, 2] with the predicted (x, y) coordinates.
        - maxvals: A 3D tensor of shape [Batch, Keypoints, 1] with the peak confidence scores.
)pbdoc",
          py::arg("heatmaps"),
          py::arg("center"),
          py::arg("scale"),
          py::arg("unbiased") = false,
          py::arg("post_process") = "default",
          py::arg("kernel") = 11,
          py::arg("valid_radius_factor") = 0.0546875f,
          py::arg("use_udp") = false,
          py::arg("target_type") = "GaussianHeatmap");
}


