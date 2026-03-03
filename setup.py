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
                extra_compile_args.append(token)

        return include_dirs, library_dirs, libraries, extra_compile_args
    except Exception:
        return [], [], [], []

# Get OpenCV paths (via pkg-config or fallback)
include_dirs, library_dirs, libraries, extra_compile_args = pkg_config_flags('opencv4')

if not include_dirs:
    if sys.platform == 'darwin':
        # macOS Homebrew fallback (Apple Silicon or Intel)
        brew_prefix = '/opt/homebrew' if os.uname().machine == 'arm64' else '/usr/local'
        if os.path.exists(os.path.join(brew_prefix, 'include', 'opencv4')):
            include_dirs = [os.path.join(brew_prefix, 'include', 'opencv4')]
            library_dirs = [os.path.join(brew_prefix, 'lib')]
    elif sys.platform == 'win32':
        # Windows fallback (assuming choco install to C:\opencv)
        opencv_dir = os.environ.get('OPENCV_DIR', r'C:\opencv\build')
        include_dirs = [os.path.join(opencv_dir, 'include')]
        lib_dir = os.path.join(opencv_dir, 'x64', 'vc16', 'lib')  # Adjust vcXX based on your MSVC version (vc15/vc16/vc17)
        library_dirs = [lib_dir]
        libraries = ['opencv_core490', 'opencv_imgproc490', 'opencv_highgui490']  # Adjust version (e.g., 490 for 4.9.0); use 'opencv_world490' if built as single lib
    else:
        # Linux fallback
        include_dirs = ['/usr/include/opencv4']
        library_dirs = ['/usr/lib64']  # Or /usr/lib

    if not libraries:
        libraries = ['opencv_core', 'opencv_imgproc', 'opencv_highgui']

# Conditionally set compiler flag based on OS/compiler
if sys.platform == 'win32':
    extra_compile_args += ['/std:c++17']
else:
    extra_compile_args += ['-std=c++17']

# Only pybind11 and OpenCV; no Torch
include_dirs += [pybind11.get_include()]

ext_modules = [
    Extension(
        name='my_keypoints.bindings',
        sources=['src/my_keypoints/bindings.cc', 'src/heatmaps_to_keypoints.cc'],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        language='c++',
        extra_compile_args=extra_compile_args,
    )
]

setup(ext_modules=ext_modules)