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
    if sys.platform == 'darwin':
        # macOS Homebrew fallback (Apple Silicon or Intel)
        brew_prefix = '/opt/homebrew' if os.uname().machine == 'arm64' else '/usr/local'
        if os.path.exists(os.path.join(brew_prefix, 'include', 'opencv4')):
            opencv_includes = [os.path.join(brew_prefix, 'include', 'opencv4')]
            opencv_libdirs = [os.path.join(brew_prefix, 'lib')]
    elif sys.platform == 'win32':
        # Windows fallback: robustly search for OpenCV headers from standard paths
        opencv_libs = ['opencv_world490'] # Default for choco Windows prebuilt
        search_roots = [os.environ.get('OPENCV_DIR', ''), r'C:\opencv', r'C:\tools\opencv']
        found = False
        for root_dir in search_roots:
            if not root_dir or not os.path.exists(root_dir):
                continue
            for root, dirs, files in os.walk(root_dir):
                if 'opencv2' in dirs and os.path.exists(os.path.join(root, 'opencv2', 'opencv.hpp')):
                    opencv_includes = [root]
                    parent_build = os.path.dirname(root)
                    for vc in ['vc17', 'vc16', 'vc15', 'vc14']:
                        pot_lib = os.path.join(parent_build, 'x64', vc, 'lib')
                        if os.path.exists(pot_lib):
                            opencv_libdirs = [pot_lib]
                            break
                    found = True
                    break
            if found:
                break
        
        # Absolute last resort Windows fallback
        if not opencv_includes:
            opencv_includes = [r'C:\opencv\build\include']
            opencv_libdirs = [r'C:\opencv\build\x64\vc16\lib']
            
    # Generic Linux fallback if nothing else matched
    if not opencv_includes:
        opencv_includes = ['/usr/include/opencv4']
        
if not opencv_libs:
    opencv_libs = ["opencv_core", "opencv_imgproc"]

# Add C++17 flag for Windows specifically if needed
if sys.platform == 'win32' and '/std:c++17' not in opencv_compile_args:
    opencv_compile_args.append('/std:c++17')

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
