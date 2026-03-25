from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension
import os

this_dir = os.path.dirname(os.path.abspath(__file__))

setup(
    name="heatmaps_to_keypoints",
    version="0.0.2",
    ext_modules=[
        CppExtension(
            name="heatmaps_to_keypoints",
            sources=[
                "keypoint_heatmap.cpp",
            ],
            include_dirs=[this_dir, "/usr/include/opencv4"],  
            libraries=["opencv_core", "opencv_imgproc"],
            extra_compile_args=["-O3"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)
