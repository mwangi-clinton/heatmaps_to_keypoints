from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension
import subprocess
import shlex
import sys
import os

this_dir = os.path.dirname(os.path.abspath(__file__))

def pkg_config_flags(package_name='opencv4'):
    try:
        cflags = subprocess.check_output(['pkg-config', '--cflags', package_name], text=True).strip()
        libs = subprocess.check_output(['pkg-config', '--libs', package_name], text=True).strip()
        include_dirs = []
        library_dirs = []
        libraries = []
        extra_compile_args = []

        for token in shlex.split(cflags):
            if token.startswith('-I'):
                include_dirs.append(token[2:])
            else:
                extra_compile_args.append(token)

        for token in shlex.split(libs):
            if token.startswith('-L'):
                library_dirs.append(token[2:])
            elif token.startswith('-l'):
                libraries.append(token[2:])
            else:
                extra_compile_args.append(token)

        return include_dirs, library_dirs, libraries, extra_compile_args
    except Exception:
        return [], [], [], []

opencv_includes, opencv_libdirs, opencv_libs, opencv_compile_args = pkg_config_flags('opencv4')

if not opencv_includes:
    opencv_includes = ["/usr/include/opencv4"]
if not opencv_libs:
    opencv_libs = ["opencv_core", "opencv_imgproc"]

setup(
    name="heatmaps_to_keypoints",
    version="0.1.0",
    ext_modules=[
        CppExtension(
            name="heatmaps_to_keypoints",
            sources=[
                "keypoint_heatmap.cpp",
            ],
            include_dirs=[this_dir] + opencv_includes,
            library_dirs=opencv_libdirs,
            libraries=opencv_libs,
            extra_compile_args=["-O3"] + opencv_compile_args,
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)
