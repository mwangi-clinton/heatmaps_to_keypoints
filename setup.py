from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension
import os

this_dir = os.path.dirname(os.path.abspath(__file__))

setup(
    name="heatmaps_to_keypoints",
    version="0.0.1",
    ext_modules=[
        CppExtension(
            name="heatmaps_to_keypoints",
            sources=[
                "decoder_bindings.cc",
                "heatmaps_to_keypoints.cc",
            ],
            include_dirs=[this_dir],  
            extra_compile_args=["-O3"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)
