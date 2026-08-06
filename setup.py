# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import glob
import os
import re
import subprocess
import sys
import warnings
from typing import Any, cast

from packaging import version
from setuptools import find_packages, setup

import versioneer

# ---------- ROCm detection (hipCIM-style) ----------

DEFAULT_ROCM_SERIES = "7.14"
DEFAULT_GPU_ARCHS = ("gfx942", "gfx950")


def _run(cmd):
    """stdout of ``cmd``, or "" if it is missing or fails."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout
    except OSError:
        return ""


def _installed_rocm_version():
    """MAJOR.MINOR from the installed rocm or rocm-sdk-core package, or ""."""
    try:
        from importlib.metadata import version as pkg_version
        ver = pkg_version("rocm")
        match = re.match(r"(\d+\.\d+)", ver)
        return match.group(1) if match else ""
    except Exception:
        return ""


def _detect_rocm_series():
    """ROCm MAJOR.MINOR for the torch[device] and rocm pin. First match wins:
    MONAI_ROCM_SERIES env var, installed rocm package, rocm-sdk CLI, hipcc,
    else DEFAULT_ROCM_SERIES.
    """
    for text, pattern in (
        (os.environ.get("MONAI_ROCM_SERIES", ""), r"(\d+\.\d+)"),
        (_installed_rocm_version(), r"(\d+\.\d+)"),
        (_run(["rocm-sdk", "version"]), r"(\d+\.\d+)"),
        (_run(["hipcc", "--version"]), r"HIP version:\s*(\d+\.\d+)"),
    ):
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return DEFAULT_ROCM_SERIES


def _detect_gpu_archs():
    """AMD GPU architectures for torch[device-gfx*] extras. Reads GPU_TARGETS /
    AMDGPU_TARGETS env vars; falls back to DEFAULT_GPU_ARCHS.
    """
    raw = os.environ.get("GPU_TARGETS") or os.environ.get("AMDGPU_TARGETS", "")
    archs = []
    for match in re.findall(r"gfx[0-9a-f]+", raw, re.IGNORECASE):
        arch = match.lower()
        if arch not in archs:
            archs.append(arch)
    return archs or list(DEFAULT_GPU_ARCHS)


def _rocm_install_requires():
    """Generate dynamic install_requires for ROCm: torch with device extras and
    a rocm version pin so the resolver picks the correct ROCm build.
    """
    rocm_series = _detect_rocm_series()
    gpu_archs = _detect_gpu_archs()
    device_extras = ",".join(f"device-{arch}" for arch in gpu_archs)
    return [
        f"torch[{device_extras}]>=2.8.0",
        f"rocm>={rocm_series}.0a0,<{int(rocm_series.split('.')[0])}.{int(rocm_series.split('.')[1]) + 1}",
    ]

# TODO: debug mode -g -O0, compile test cases

RUN_BUILD = os.getenv("BUILD_MONAI", "0") == "1"
FORCE_CUDA = os.getenv("FORCE_CUDA", "0") == "1"  # flag ignored if BUILD_MONAI is False

BUILD_CPP = BUILD_CUDA = False
TORCH_VERSION = 0
try:
    import torch

    print(f"setup.py with torch {torch.__version__}")
    from torch.utils.cpp_extension import BuildExtension, CppExtension

    BUILD_CPP = True
    from torch.utils.cpp_extension import CUDA_HOME, CUDAExtension

    BUILD_CUDA = FORCE_CUDA or (torch.cuda.is_available() and (CUDA_HOME is not None))

    _pt_version = version.parse(torch.__version__).release
    if _pt_version is None or len(_pt_version) < 3:
        raise AssertionError("unknown torch version")
    TORCH_VERSION = int(_pt_version[0]) * 10000 + int(_pt_version[1]) * 100 + int(_pt_version[2])
except (ImportError, TypeError, AssertionError, AttributeError) as e:
    if RUN_BUILD:
        raise RuntimeError(
            f"torch is required to build MONAI C extensions but could not be imported: {e}\n"
            "Install ROCm torch first, then build with: pip install --no-build-isolation -e ."
        ) from e
    warnings.warn(f"extension build skipped: {e}")
finally:
    if not RUN_BUILD:
        BUILD_CPP = BUILD_CUDA = False
        print("Please set environment variable `BUILD_MONAI=1` to enable Cpp/CUDA extension build.")
    print(f"BUILD_MONAI_CPP={BUILD_CPP}, BUILD_MONAI_CUDA={BUILD_CUDA}, TORCH_VERSION={TORCH_VERSION}.")


def torch_parallel_backend():
    try:
        match = re.search("^ATen parallel backend: (?P<backend>.*)$", torch._C._parallel_info(), re.MULTILINE)
        if match is None:
            return None
        backend = match.group("backend")
        if backend == "OpenMP":
            return "AT_PARALLEL_OPENMP"
        if backend == "native thread pool":
            return "AT_PARALLEL_NATIVE"
        if backend == "native thread pool and TBB":
            return "AT_PARALLEL_NATIVE_TBB"
    except (NameError, AttributeError):  # no torch or no binaries
        warnings.warn("Could not determine torch parallel_info.")
    return None


def omp_flags():
    if sys.platform == "win32":
        return ["/openmp"]
    if sys.platform == "darwin":
        # https://stackoverflow.com/questions/37362414/
        # return ["-fopenmp=libiomp5"]
        return []
    return ["-fopenmp"]


def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    ext_dir = os.path.join(this_dir, "monai", "csrc")
    include_dirs = [ext_dir]

    source_cpu = glob.glob(os.path.join(ext_dir, "**", "*.cpp"), recursive=True)
    source_cuda = glob.glob(os.path.join(ext_dir, "**", "*.cu"), recursive=True)

    extension = None
    define_macros = [(f"{torch_parallel_backend()}", 1), ("MONAI_TORCH_VERSION", TORCH_VERSION)]
    extra_compile_args = {}
    extra_link_args = []
    sources = source_cpu
    if BUILD_CPP:
        extension = CppExtension
        extra_compile_args.setdefault("cxx", [])
        if torch_parallel_backend() == "AT_PARALLEL_OPENMP":
            extra_compile_args["cxx"] += omp_flags()
        extra_link_args = omp_flags()
    if BUILD_CUDA:
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        # Embed the maximum compute capability from TORCH_CUDA_ARCH_LIST
        _torch_cuda_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", "")
        _max_cc = 0
        if _torch_cuda_arch_list:
            for _maj, _min in re.findall(r"([0-9]+)\.([0-9]+)", _torch_cuda_arch_list):
                try:
                    _cc = int(_maj) * 100 + int(_min)
                    if _cc > _max_cc:
                        _max_cc = _cc
                except ValueError:
                    pass
        if _max_cc > 0:
            define_macros += [("MONAI_MAX_COMPUTE_CAPABILITY", _max_cc)]
        extra_compile_args = {"cxx": [], "nvcc": []}
        if torch_parallel_backend() == "AT_PARALLEL_OPENMP":
            extra_compile_args["cxx"] += omp_flags()
    if extension is None or not sources:
        return []  # compile nothing

    ext_modules = [
        extension(
            name="monai._C",
            sources=sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    ]
    return ext_modules


def get_cmds():
    cmds = versioneer.get_cmdclass()

    if not (BUILD_CPP or BUILD_CUDA):
        return cmds

    cmds.update({"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)})
    return cmds


# Gathering source used for JIT extensions to include in package_data.
jit_extension_source = []

for ext in ["cpp", "cu", "h", "cuh"]:
    glob_path = os.path.join("monai", "_extensions", "**", f"*.{ext}")
    jit_extension_source += glob.glob(glob_path, recursive=True)

jit_extension_source = [os.path.join("..", path) for path in jit_extension_source]

setup(
    version=versioneer.get_version(),
    cmdclass=get_cmds(),
    packages=find_packages(exclude=("docs", "examples", "tests", "tests.*")),
    zip_safe=False,
    package_data=cast(Any, {"monai": ["py.typed", *jit_extension_source]}),
    ext_modules=get_extensions(),
    install_requires=_rocm_install_requires() + ["numpy>=1.24,<2.5"],
)
