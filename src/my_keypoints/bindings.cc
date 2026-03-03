#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // For std::vector, std::array
#include <pybind11/numpy.h>
#include "../heatmaps_to_keypoints.h" // Include declarations for the implementation

namespace py = pybind11;

PYBIND11_MODULE(bindings, m)
{ // 'bindings' is the module name (will be installed as my_keypoints.bindings)
    m.doc() = "Python bindings for keypoints from heatmaps";

    // Bind structs (if needed)
    py::class_<CentersAndScales>(m, "CentersAndScales")
        .def(py::init<>())
        .def_readwrite("centers", &CentersAndScales::centers)
        .def_readwrite("scales", &CentersAndScales::scales);

    py::class_<KeypointsResult>(m, "KeypointsResult")
        .def(py::init<>())
        .def_readwrite("preds", &KeypointsResult::preds)
        .def_readwrite("maxvals", &KeypointsResult::maxvals);

    // Bind functions
    m.def("get_centers_and_scales_xyxy", &get_centers_and_scales_xyxy,
          py::arg("person_boxes"), py::arg("scale_factor") = 1.0f);

    // Original binding that expects a fully C++-typed Heatmaps4D (useful when
    // calling from C++/native objects). For Python callers who have NumPy
    // arrays, use the wrapper below.
    m.def("keypoints_from_heatmaps_with_boxes", &keypoints_from_heatmaps_with_boxes,
          py::arg("heatmaps"), py::arg("boxes"), py::arg("scale_factor") = 1.0f,
          py::arg("unbiased") = false, py::arg("post_process") = "default",
          py::arg("kernel") = 11, py::arg("valid_radius_factor") = 0.0546875f,
          py::arg("use_udp") = false, py::arg("target_type") = "GaussianHeatmap");

    // NumPy-friendly wrapper: accept a float32 NumPy array with shape (N, K, H, W)
    // and convert it into the Heatmaps4D structure (vector<vector<cv::Mat>>)
    m.def("keypoints_from_heatmaps_with_boxes_np", [](py::array_t<float, py::array::c_style | py::array::forcecast> arr, const std::vector<Box> &boxes, float scale_factor = 1.0f, bool unbiased = false, std::string post_process = "default", int kernel = 11, float valid_radius_factor = 0.0546875f, bool use_udp = false, std::string target_type = "GaussianHeatmap")
          {
              // Validate array
              if (arr.ndim() != 4)
                  throw std::runtime_error("heatmaps must be a 4D numpy array (N,K,H,W)");
              auto buf = arr.request();
              ssize_t N = buf.shape[0];
              ssize_t K = buf.shape[1];
              ssize_t H = buf.shape[2];
              ssize_t W = buf.shape[3];

              // Create Heatmaps4D
              Heatmaps4D heatmaps(static_cast<size_t>(N), Heatmaps3D(static_cast<size_t>(K)));

              const float *data = static_cast<const float *>(buf.ptr);
              // row-major: index = ((n*K + k)*H + h)*W + w
              for (ssize_t n = 0; n < N; ++n)
              {
                  for (ssize_t k = 0; k < K; ++k)
                  {
                      Mat mat(static_cast<int>(H), static_cast<int>(W), CV_32F);
                      for (ssize_t h = 0; h < H; ++h)
                      {
                          float *row_ptr = mat.ptr<float>(static_cast<int>(h));
                          const float *src = data + ((n * K + k) * H + h) * W;
                          std::copy(src, src + W, row_ptr);
                      }
                      heatmaps[static_cast<size_t>(n)][static_cast<size_t>(k)] = mat;
                  }
              }

              return keypoints_from_heatmaps_with_boxes(heatmaps, boxes, scale_factor,
                                                       unbiased, post_process, kernel,
                                                       valid_radius_factor, use_udp,
                                                       target_type); }, py::arg("heatmaps"), py::arg("boxes"), py::arg("scale_factor") = 1.0f, py::arg("unbiased") = false, py::arg("post_process") = "default", py::arg("kernel") = 11, py::arg("valid_radius_factor") = 0.0546875f, py::arg("use_udp") = false, py::arg("target_type") = "GaussianHeatmap");

    // Bind other internal functions if you want them exposed
}