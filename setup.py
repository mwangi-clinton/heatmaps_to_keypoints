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
        # Windows fallback: robustly search for OpenCV headers from standard paths
        include_dirs = []
        library_dirs = []
        libraries = ['opencv_world490'] # Default for choco Windows prebuilt

        search_roots = [os.environ.get('OPENCV_DIR', ''), r'C:\opencv', r'C:\tools\opencv']
        found = False
        for root_dir in search_roots:
            if not root_dir or not os.path.exists(root_dir):
                continue
            for root, dirs, files in os.walk(root_dir):
                if 'opencv2' in dirs and os.path.exists(os.path.join(root, 'opencv2', 'opencv.hpp')):
                    include_dirs = [root]
                    parent_build = os.path.dirname(root)
                    # Try to find lib dir
                    for vc in ['vc17', 'vc16', 'vc15', 'vc14']:
                        pot_lib = os.path.join(parent_build, 'x64', vc, 'lib')
                        if os.path.exists(pot_lib):
                            library_dirs = [pot_lib]
                            break
                    found = True
                    break
            if found:
                break

        # Fallback if nothing found
        if not include_dirs:
            include_dirs = [r'C:\opencv\build\include']
            library_dirs = [r'C:\opencv\build\x64\vc16\lib']
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