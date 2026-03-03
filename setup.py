from setuptools import setup, Extension
import pybind11
import subprocess
import shlex
import sys
import os

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
                # pass through other flags
                extra_compile_args.append(token)

        return include_dirs, library_dirs, libraries, extra_compile_args
    except Exception:
        return [], [], [], []

# Try to detect OpenCV via pkg-config, fall back to common locations
include_dirs, library_dirs, libraries, extra_compile_args = pkg_config_flags('opencv4')
if not include_dirs:
    # Common Homebrew path on macOS Apple Silicon
    brew_prefix = '/opt/homebrew'
    if sys.platform == 'darwin' and os.path.exists(os.path.join(brew_prefix, 'include', 'opencv4')):
        include_dirs = [os.path.join(brew_prefix, 'include', 'opencv4')]
        library_dirs = [os.path.join(brew_prefix, 'lib')]
    else:
        include_dirs = ['/usr/include/opencv4']
        library_dirs = ['/usr/lib']
    libraries = ['opencv_core', 'opencv_imgproc', 'opencv_highgui']

ext_modules = [
    Extension(
        name='my_keypoints.bindings',
        sources=['src/my_keypoints/bindings.cc', 'src/heatmaps_to_keypoints.cc'],
        include_dirs=[pybind11.get_include()] + include_dirs,
        libraries=libraries,
        library_dirs=library_dirs,
        language='c++',
        extra_compile_args=['-std=c++17'] + extra_compile_args,
    )
]

setup(ext_modules=ext_modules)
