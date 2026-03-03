from setuptools import setup, Extension
import pybind11
import sys

opencv_include = "/usr/include/opencv4"
opencv_libs = ["opencv_core", "opencv_imgproc", "opencv_highgui"]

ext_modules = [
    Extension(
        name="my_keypoints.bindings",
        sources=["src/my_keypoints/bindings.cc", "src/heatmaps_to_keypoints.cc"],
        include_dirs=[pybind11.get_include(), opencv_include],
        libraries=opencv_libs,
        library_dirs=["/usr/lib"],
        language="c++",
        extra_compile_args=["-std=c++17"],
    )
]

setup(ext_modules=ext_modules)
