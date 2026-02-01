from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension
import os
import sys

this_dir = os.path.dirname(os.path.abspath(__file__))

setup(
    name="heatmaps_to_keypoints",
    version="0.0.4",
    ext_modules=[
        CppExtension(
            name="heatmaps_to_keypoints",
            sources=[
                "decoder_bindings.cc",
                "heatmaps_to_keypoints.cc",
            ],
            include_dirs=[this_dir],  
            extra_compile_args=["-O3"],
            define_macros=[('_GLIBCXX_USE_CXX11_ABI', '0')],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)
