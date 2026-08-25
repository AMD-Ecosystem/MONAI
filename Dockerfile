# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ROCm 7.14 base with PyTorch and Python 3.12 preinstalled. This is a TheRock
# image: ROCm ships as pip packages under the venv's _rocm_sdk_core (no /opt/rocm),
# and torch/torchvision/torchaudio are already installed. To install PyTorch on a
# different base, see:
#   https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html
ARG BASE_IMAGE=rocm/pytorch:rocm7.14_ubuntu24.04_py3.12_pytorch_release_2.12.0
FROM ${BASE_IMAGE}

# Base already provides build-essential, cmake, ninja, git and gcc-13. Reinstall
# build-essential explicitly (the Triton AMD backend JIT-compiles HIP kernels at
# runtime and needs the toolchain) and add the imaging dev libraries MONAI needs.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential git cmake ninja-build yasm \
    openssh-client \
    libopenslide-dev libwebp-dev libzstd-dev && \
    rm -rf /var/lib/apt/lists/*

# ROCm is a pip package in this base (no /opt/rocm). Point the build at the pip
# SDK so torch's HIP extension compiler (hipcc) and the HIP headers are found,
# and expose its lib dir at runtime.
ENV ROCM_PATH="/opt/venv/lib/python3.12/site-packages/_rocm_sdk_core"
ENV ROCM_HOME="${ROCM_PATH}"
ENV ROCM_DEVEL_PATH="/opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel"
ENV ROCM_LIBRARIES_PATH="/opt/venv/lib/python3.12/site-packages/_rocm_sdk_libraries"
ENV PATH="${ROCM_PATH}/bin:/opt/venv/bin:${PATH}"
# TheRock splits runtime libs across core + libraries (+ nested subdirs); the
# math libs (libhipblas etc.) that cupy loads live in the libraries tree.
ENV LD_LIBRARY_PATH="${ROCM_PATH}/lib:${ROCM_PATH}/lib/rocm_sysdeps/lib:${ROCM_PATH}/lib/llvm/lib:${ROCM_LIBRARIES_PATH}/lib:${LD_LIBRARY_PATH}"
# Link time (unlike LD_LIBRARY_PATH) resolves via LIBRARY_PATH. The unversioned
# lib*.so dev symlinks (e.g. libamdhip64.so) only exist in the devel tree.
ENV LIBRARY_PATH="${ROCM_DEVEL_PATH}/lib:${ROCM_PATH}/lib"

# clang locates ROCm device bitcode at ${ROCM_PATH}/amdgcn/bitcode, but this
# TheRock SDK ships it under lib/llvm/amdgcn/bitcode. Once ROCM_PATH is set,
# clang trusts it and fails the HIP device-library lookup without this symlink.
RUN mkdir -p "${ROCM_PATH}/amdgcn" && \
    ln -sf "${ROCM_PATH}/lib/llvm/amdgcn/bitcode" "${ROCM_PATH}/amdgcn/bitcode"

# GPU arch(s) for the from-source HIP device codegen (overridable via --build-arg).
ARG AMDGPU_TARGETS="gfx942;gfx950"
ENV AMDGPU_TARGETS=${AMDGPU_TARGETS}
# torch's HIP extension build reads PYTORCH_ROCM_ARCH (not AMDGPU_TARGETS) to emit
# --offload-arch. Unset, a GPU-less build container yields an empty arch and the
# HIP device-library lookup fails, so pin it to the same targets.
ENV PYTORCH_ROCM_ARCH=${AMDGPU_TARGETS}

# AMD PyPI index for amd-hipcim and amd-monai wheels. Torch is served from a
# separate index (repo.amd.com/rocm/whl-multi-arch) and is already installed in
# the base image, so this ARG only governs AMD-built Python packages.
ARG AMD_PIP_INDEX="https://pypi.amd.com/rocm-10.0.0/simple/"

# PyTorch is preinstalled in the base venv — do not install it here.
# Install amd-hipcim (digital-pathology I/O; provides the `cucim` module) and the
# MONAI development requirements. hipCIM is pulled from the AMD pip index; the
# import guard fails the build loudly if only the name-reservation stub resolves.
COPY ./requirements*.txt /tmp/
COPY ./amd-constraints.txt /tmp/

# Expose the ROCm devel headers (rocThrust/rocPRIM/hipCUB via _rocm_sdk_devel) to
# the HIP compile, plus the gcc-13 include path workaround for CuPy.
ENV CPATH="/opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel/include:/usr/lib/gcc/x86_64-linux-gnu/13/include"

RUN pip install --no-cache-dir --upgrade pip wheel && \
    pip install --no-cache-dir amd-hipcim --extra-index-url="${AMD_PIP_INDEX}" && \
    python3 -c "import cucim; print('hipCIM (cucim) OK:', cucim.__version__)" && \
    pip install --no-cache-dir -r /tmp/requirements-dev.txt -c /tmp/amd-constraints.txt

# The runtime base ships no ROCm devel headers (rocThrust/rocPRIM/hipCUB), which
# the from-source HIP compile needs. rocm-sdk-devel comes from the ROCm/torch
# wheel index (repo.amd.com/rocm/whl-multi-arch), not from AMD_PIP_INDEX which
# hosts amd-monai/amd-hipcim. rocm-sdk init expands the devel tree on disk.
RUN pip install --no-cache-dir "rocm-sdk-devel==7.14.0" --extra-index-url=https://repo.amd.com/rocm/whl-multi-arch/ && \
    rocm-sdk init

COPY . /monai

WORKDIR /monai

RUN git config --global --add safe.directory /monai

# Build MONAI from source (editable install). Use --no-build-isolation so the
# build sees the base image's pre-installed ROCm torch; isolation would pull a
# CUDA torch into the build env and fail with "CUDA_HOME not set". more-itertools
# is a build-system requirement not shipped in the base venv.
RUN pip install --no-cache-dir more-itertools && \
    BUILD_MONAI=1 FORCE_CUDA=1 pip install --no-build-isolation -e .

RUN python3 -c "import torch; print('Torch version:', torch.__version__)" && \
    python3 -c "import cupy; print('amd cupy version:', cupy.__version__)" && \
    python3 -c "import monai; print('monai version:', monai.__version__)" && \
    python3 -c "import cucim; print('hipcim version:', cucim.__version__)"
