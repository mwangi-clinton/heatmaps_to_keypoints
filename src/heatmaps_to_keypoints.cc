// Standard includes used by the implementation
#include <algorithm> // for std::swap, std::min, std::max
#include <cmath>     // for std::floor, std::abs
#include <cstddef>   // for std::size_t
#include <stdexcept> // for std::runtime_error
#include <numeric>   // for std::numeric_limits
#include <iostream>  // for output in main

#include "heatmaps_to_keypoints.h"

/**
 * Compute centers and scales for person bounding boxes in XYXY format.
 *
 * @param person_boxes  Vector of boxes, each box is {x1, y1, x2, y2}
 * @param scale_factor  Optional enlargement factor (default = 1.0)
 *                      e.g. 1.25 for some padding around the person
 * @return struct containing two vectors: centers and scales
 */
CentersAndScales get_centers_and_scales_xyxy(
    const std::vector<Box> &person_boxes,
    float scale_factor)
{
  Centers centers;
  Scales scales;

  // Reserve memory to avoid repeated reallocations
  centers.reserve(person_boxes.size());
  scales.reserve(person_boxes.size());

  for (const auto &box : person_boxes)
  {
    float x1 = box[0];
    float y1 = box[1];
    float x2 = box[2];
    float y2 = box[3];

    // Make sure x1 < x2 and y1 < y2 (same as Python's sorted)
    if (x1 > x2)
      std::swap(x1, x2);
    if (y1 > y2)
      std::swap(y1, y2);

    float center_x = (x1 + x2) * 0.5f;
    float center_y = (y1 + y2) * 0.5f;

    centers.push_back({center_x, center_y});

    float w = x2 - x1;
    float h = y2 - y1;

    float scale_x = (w / 200.0f) * scale_factor;
    float scale_y = (h / 200.0f) * scale_factor;

    scales.push_back({scale_x, scale_y});
  }

  return {std::move(centers), std::move(scales)};
}

std::pair<Coords3D, Maxvals3D> _get_max_preds(const Heatmaps4D &heatmaps)
{
  size_t N = heatmaps.size();
  if (N == 0)
    return {{}, {}};

  size_t K = heatmaps[0].size();
  int H = heatmaps[0][0].rows;
  int W = heatmaps[0][0].cols;

  Coords3D preds(N, Coords2D(K, {-1.f, -1.f}));
  Maxvals3D maxvals(N, Maxvals2D(K, 0.f));

  for (size_t n = 0; n < N; ++n)
  {
    for (size_t k = 0; k < K; ++k)
    {
      const Mat &hm = heatmaps[n][k];
      double maxv;
      Point max_loc;
      minMaxLoc(hm, nullptr, &maxv, nullptr, &max_loc);
      maxvals[n][k] = static_cast<float>(maxv);
      if (maxv > 0.0)
      {
        preds[n][k][0] = static_cast<float>(max_loc.x);
        preds[n][k][1] = static_cast<float>(max_loc.y);
      } // else remains -1
    }
  }
  return {preds, maxvals};
}

void _gaussian_blur(Heatmaps4D &heatmaps, int kernel = 11)
{
  if (kernel % 2 != 1)
  {
    throw std::runtime_error("Kernel must be odd");
  }
  int border = (kernel - 1) / 2;
  size_t N = heatmaps.size();
  if (N == 0)
    return;
  size_t K = heatmaps[0].size();
  int H = heatmaps[0][0].rows;
  int W = heatmaps[0][0].cols;

  for (size_t i = 0; i < N; ++i)
  {
    for (size_t j = 0; j < K; ++j)
    {
      Mat &hm = heatmaps[i][j];
      double origin_max;
      minMaxLoc(hm, nullptr, &origin_max, nullptr, nullptr);

      Mat dr = Mat::zeros(H + 2 * border, W + 2 * border, CV_32F);
      hm.copyTo(dr(Rect(border, border, W, H)));
      GaussianBlur(dr, dr, Size(kernel, kernel), 0);
      dr(Rect(border, border, W, H)).copyTo(hm);

      double new_max;
      minMaxLoc(hm, nullptr, &new_max, nullptr, nullptr);
      if (new_max > 0.0)
      {
        hm *= static_cast<float>(origin_max / new_max);
      }
    }
  }
}

Coords3D post_dark_udp(Coords3D coords, Heatmaps4D batch_heatmaps, int kernel = 3)
{
  size_t N = coords.size();
  if (N == 0)
    return coords;
  size_t K = coords[0].size();
  size_t B = batch_heatmaps.size();
  int H = batch_heatmaps[0][0].rows;
  int W = batch_heatmaps[0][0].cols;
  if (B != 1 && B != N)
  {
    throw std::runtime_error("B must be 1 or N");
  }

  // Gaussian blur each heatmap
  for (size_t b = 0; b < B; ++b)
  {
    for (size_t k = 0; k < K; ++k)
    {
      GaussianBlur(batch_heatmaps[b][k], batch_heatmaps[b][k], Size(kernel, kernel), 0);
    }
  }

  // Clip to [0.001, 50]
  for (auto &jh : batch_heatmaps)
  {
    for (auto &hm : jh)
    {
      max(hm, 0.001f, hm); // Replace <0.001 with 0.001
      min(hm, 50.f, hm);
    }
  }

  // Log
  for (auto &jh : batch_heatmaps)
  {
    for (auto &hm : jh)
    {
      log(hm, hm);
    }
  }

  // Pad and flatten (edge mode)
  int pad_H = H + 2;
  int pad_W = W + 2;
  std::vector<float> batch_heatmaps_pad(B * K * pad_H * pad_W);
  for (size_t b = 0; b < B; ++b)
  {
    for (size_t k = 0; k < K; ++k)
    {
      Mat padded;
      copyMakeBorder(batch_heatmaps[b][k], padded, 1, 1, 1, 1, BORDER_REPLICATE);
      size_t offset = (b * K + k) * pad_H * pad_W;
      for (int r = 0; r < pad_H; ++r)
      {
        const float *src = padded.ptr<float>(r);
        std::copy(src, src + pad_W, batch_heatmaps_pad.begin() + offset + r * pad_W);
      }
    }
  }

  // Compute derivatives
  std::vector<float> dx(N * K);
  std::vector<float> dy(N * K);
  std::vector<float> dxx(N * K);
  std::vector<float> dyy(N * K);
  std::vector<float> dxy(N * K);
  for (size_t n = 0; n < N; ++n)
  {
    size_t bb = (B == 1 ? 0 : n);
    for (size_t k = 0; k < K; ++k)
    {
      size_t flat_idx = n * K + k;
      size_t x = static_cast<size_t>(coords[n][k][0]);
      size_t y = static_cast<size_t>(coords[n][k][1]);
      size_t index = x + 1 + (y + 1) * static_cast<size_t>(pad_W);
      size_t offset = (bb * K + k) * static_cast<size_t>(pad_H * pad_W);
      index += offset;

      float i_ = batch_heatmaps_pad[index];
      float ix1 = batch_heatmaps_pad[index + 1];
      float iy1 = batch_heatmaps_pad[index + pad_W];
      float ix1y1 = batch_heatmaps_pad[index + pad_W + 1];
      float ix1_y1_ = batch_heatmaps_pad[index - pad_W - 1];
      float ix1_ = batch_heatmaps_pad[index - 1];
      float iy1_ = batch_heatmaps_pad[index - pad_W];

      dx[flat_idx] = 0.5f * (ix1 - ix1_);
      dy[flat_idx] = 0.5f * (iy1 - iy1_);
      dxx[flat_idx] = ix1 - 2.f * i_ + ix1_;
      dyy[flat_idx] = iy1 - 2.f * i_ + iy1_;
      dxy[flat_idx] = 0.5f * (ix1y1 - ix1 - iy1 + i_ + i_ - ix1_ - iy1_ + ix1_y1_);
    }
  }

  // Hessian inverse and update coords
  float eps = std::numeric_limits<float>::epsilon();
  for (size_t n = 0; n < N; ++n)
  {
    for (size_t k = 0; k < K; ++k)
    {
      size_t idx = n * K + k;
      float dxx_ = dxx[idx];
      float dyy_ = dyy[idx];
      float dxy_ = dxy[idx];
      float det = dxx_ * dyy_ - dxy_ * dxy_;
      if (std::abs(det) <= eps)
        continue;
      float inv00 = dyy_ / det;
      float inv01 = -dxy_ / det;
      float inv10 = -dxy_ / det;
      float inv11 = dxx_ / det;
      float der_x = dx[idx];
      float der_y = dy[idx];
      float offset_x = inv00 * der_x + inv01 * der_y;
      float offset_y = inv10 * der_x + inv11 * der_y;
      coords[n][k][0] -= offset_x;
      coords[n][k][1] -= offset_y;
    }
  }
  return coords;
}

Vec2 _taylor(const Mat &heatmap, Vec2 coord)
{
  int H = heatmap.rows;
  int W = heatmap.cols;
  int px = static_cast<int>(coord[0]);
  int py = static_cast<int>(coord[1]);
  if (1 < px && px < W - 2 && 1 < py && py < H - 2)
  {
    float dx = 0.5f * (heatmap.at<float>(py, px + 1) - heatmap.at<float>(py, px - 1));
    float dy = 0.5f * (heatmap.at<float>(py + 1, px) - heatmap.at<float>(py - 1, px));
    float dxx = 0.25f * (heatmap.at<float>(py, px + 2) - 2.f * heatmap.at<float>(py, px) + heatmap.at<float>(py, px - 2));
    float dyy = 0.25f * (heatmap.at<float>(py + 2, px) - 2.f * heatmap.at<float>(py, px) + heatmap.at<float>(py - 2, px));
    float dxy = 0.25f * (heatmap.at<float>(py + 1, px + 1) - heatmap.at<float>(py - 1, px + 1) -
                         heatmap.at<float>(py + 1, px - 1) + heatmap.at<float>(py - 1, px - 1));
    float det = dxx * dyy - dxy * dxy;
    if (std::abs(det) > 1e-6f)
    {
      float inv00 = dyy / det;
      float inv01 = -dxy / det;
      float inv10 = -dxy / det;
      float inv11 = dxx / det;
      float offset_x = -(inv00 * dx + inv01 * dy);
      float offset_y = -(inv10 * dx + inv11 * dy);
      coord[0] += offset_x;
      coord[1] += offset_y;
    }
  }
  return coord;
}

Coords2D transform_preds(
    const Coords2D &coords,
    Vec2 center,
    Vec2 scale,
    Vec2 output_size,
    bool use_udp = false)
{
  scale[0] *= 200.f;
  scale[1] *= 200.f;
  float scale_x = use_udp ? scale[0] / (output_size[0] - 1.f) : scale[0] / output_size[0];
  float scale_y = use_udp ? scale[1] / (output_size[1] - 1.f) : scale[1] / output_size[1];

  Coords2D target_coords;
  target_coords.reserve(coords.size());
  for (const auto &c : coords)
  {
    float x = c[0] * scale_x + center[0] - scale[0] * 0.5f;
    float y = c[1] * scale_y + center[1] - scale[1] * 0.5f;
    target_coords.push_back({x, y});
  }
  return target_coords;
}

KeypointsResult keypoints_from_heatmaps(
    Heatmaps4D heatmaps,
    const Centers &center,
    const Scales &scale,
    bool unbiased,
    std::string post_process,
    int kernel,
    float valid_radius_factor,
    bool use_udp,
    std::string target_type)
{
  size_t N = heatmaps.size();
  if (N == 0 || center.size() != N || scale.size() != N)
  {
    throw std::runtime_error("Input sizes mismatch");
  }
  size_t K = heatmaps[0].size();
  int H = heatmaps[0][0].rows;
  int W = heatmaps[0][0].cols;

  Heatmaps4D hm = heatmaps; // Shallow copy (Mats are ref-counted)

  // Normalize post_process (handling deprecations simplistically)
  if (unbiased)
    post_process = "unbiased";

  // Start processing
  if (post_process == "megvii")
  {
    _gaussian_blur(hm, kernel);
  }

  Coords3D preds(N, Coords2D(K));
  Maxvals3D maxvals(N, Maxvals2D(K));

  if (use_udp)
  {
    std::string tt = target_type;
    std::transform(tt.begin(), tt.end(), tt.begin(), ::tolower);
    if (tt == "gaussianheatmap")
    {
      auto pm = _get_max_preds(hm);
      preds = pm.first;
      maxvals = pm.second;
      preds = post_dark_udp(preds, hm, kernel);
    }
    else if (tt == "combinedtarget")
    {
      if (K % 3 != 0)
        throw std::runtime_error("K must be divisible by 3 for CombinedTarget");
      size_t num_groups = K / 3;

      // Variable kernel blur
      for (size_t n = 0; n < N; ++n)
      {
        for (size_t i = 0; i < K; ++i)
        {
          int kt = (i % 3 == 0) ? 2 * kernel + 1 : kernel;
          GaussianBlur(hm[n][i], hm[n][i], Size(kt, kt), 0);
        }
      }

      float valid_radius = valid_radius_factor * static_cast<float>(H);

      // Extract offsets
      std::vector<float> offset_x(N * num_groups * H * W);
      std::vector<float> offset_y(N * num_groups * H * W);
      for (size_t n = 0; n < N; ++n)
      {
        for (size_t g = 0; g < num_groups; ++g)
        {
          size_t idx_off = n * num_groups * H * W + g * H * W;
          const Mat &off_x_hm = hm[n][g * 3 + 1];
          const float *src_x = off_x_hm.ptr<float>(0);
          for (int p = 0; p < H * W; ++p)
          {
            offset_x[idx_off + p] = src_x[p] * valid_radius;
          }
          const Mat &off_y_hm = hm[n][g * 3 + 2];
          const float *src_y = off_y_hm.ptr<float>(0);
          for (int p = 0; p < H * W; ++p)
          {
            offset_y[idx_off + p] = src_y[p] * valid_radius;
          }
        }
      }

      // Classification heatmaps only
      Heatmaps4D class_hm(N, Heatmaps3D(num_groups));
      for (size_t n = 0; n < N; ++n)
      {
        for (size_t g = 0; g < num_groups; ++g)
        {
          class_hm[n][g] = hm[n][g * 3];
        }
      }

      auto pm = _get_max_preds(class_hm);
      preds = pm.first;    // Now N x num_groups x 2
      maxvals = pm.second; // N x num_groups

      // Add offsets using index
      for (size_t n = 0; n < N; ++n)
      {
        for (size_t g = 0; g < num_groups; ++g)
        {
          float px = preds[n][g][0];
          float py = preds[n][g][1];
          size_t flat = static_cast<size_t>(px) + static_cast<size_t>(py) * static_cast<size_t>(W);
          size_t offset_idx = n * num_groups * static_cast<size_t>(H * W) + g * static_cast<size_t>(H * W) + flat;
          preds[n][g][0] += offset_x[offset_idx];
          preds[n][g][1] += offset_y[offset_idx];
        }
      }
    }
    else
    {
      throw std::runtime_error("Invalid target_type");
    }
  }
  else
  {
    auto pm = _get_max_preds(hm);
    preds = pm.first;
    maxvals = pm.second;
    if (post_process == "unbiased")
    {
      _gaussian_blur(hm, kernel);
      for (size_t n = 0; n < N; ++n)
      {
        for (size_t k = 0; k < K; ++k)
        {
          max(hm[n][k], 1e-10f, hm[n][k]);
          log(hm[n][k], hm[n][k]);
          preds[n][k] = _taylor(hm[n][k], preds[n][k]);
        }
      }
    }
    else if (!post_process.empty())
    {
      for (size_t n = 0; n < N; ++n)
      {
        for (size_t k = 0; k < K; ++k)
        {
          float px = preds[n][k][0];
          float py = preds[n][k][1];
          int ipx = static_cast<int>(px);
          int ipy = static_cast<int>(py);
          if (1 < ipx && ipx < W - 1 && 1 < ipy && ipy < H - 1)
          {
            float diff_x = hm[n][k].at<float>(ipy, ipx + 1) - hm[n][k].at<float>(ipy, ipx - 1);
            float diff_y = hm[n][k].at<float>(ipy + 1, ipx) - hm[n][k].at<float>(ipy - 1, ipx);
            preds[n][k][0] += (diff_x > 0.f ? 0.25f : (diff_x < 0.f ? -0.25f : 0.f));
            preds[n][k][1] += (diff_y > 0.f ? 0.25f : (diff_y < 0.f ? -0.25f : 0.f));
            if (post_process == "megvii")
            {
              preds[n][k][0] += 0.5f;
              preds[n][k][1] += 0.5f;
            }
          }
        }
      }
    }
  }

  // Transform back
  Vec2 output_size = {static_cast<float>(W), static_cast<float>(H)};
  for (size_t i = 0; i < N; ++i)
  {
    preds[i] = transform_preds(preds[i], center[i], scale[i], output_size, use_udp);
  }

  if (post_process == "megvii")
  {
    for (size_t n = 0; n < N; ++n)
    {
      for (size_t k = 0; k < K; ++k)
      {
        maxvals[n][k] /= 255.f;
        maxvals[n][k] += 0.5f;
      }
    }
  }

  return {preds, maxvals};
}

// Wrapper to input boxes from detection (computes centers and scales internally)
KeypointsResult keypoints_from_heatmaps_with_boxes(
    Heatmaps4D heatmaps,
    const std::vector<Box> &boxes,
    float scale_factor,
    bool unbiased,
    std::string post_process,
    int kernel,
    float valid_radius_factor,
    bool use_udp,
    std::string target_type)
{
  auto cs = get_centers_and_scales_xyxy(boxes, scale_factor);
  return keypoints_from_heatmaps(
      heatmaps, cs.centers, cs.scales, unbiased, post_process, kernel,
      valid_radius_factor, use_udp, target_type);
}
