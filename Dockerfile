# syntax=docker/dockerfile:1
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

FROM ubuntu:24.04

# Triton AMD backend JIT-compiles hip_utils at runtime — needs build tools.
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        openssh-client \
        python3-venv \
        python3-pip \
        python3-dev \
        build-essential \
        cmake \
        ninja-build \
        yasm \
        libopenslide-dev \
        libwebp-dev \
        libzstd-dev \
        libstdc++-13-dev \
    && rm -rf /var/lib/apt/lists/*


RUN python3 -m venv /opt/monai-venv
ENV PATH="/opt/monai-venv/bin:${PATH}"
RUN pip install --no-cache-dir --upgrade pip wheel

COPY . /src/monai
WORKDIR /src/monai

ARG AMDGPU_TARGETS="gfx942;gfx950"
ENV AMDGPU_TARGETS=${AMDGPU_TARGETS}

ENV PYTORCH_INDEX_URL="https://stable.repo.amd.com/rocm/whl-next/"
ENV DEP_INDEX_URL="https://pypi.amd.com/rocm-10.0.0/simple/"
# Install ROCm SDK (with devel headers), torch, and hipCIM together to ensure
# consistent ROCm version resolution across all packages.
RUN pip install --no-cache-dir --index-url ${PYTORCH_INDEX_URL} \
        "rocm[libraries,devel,device-gfx942,device-gfx950]==10.0.*" \
        "torch[device-gfx942,device-gfx950]" \
        "torchvision[device-gfx942,device-gfx950]" \
        torchaudio \
    && pip install --no-cache-dir \
        --extra-index-url ${DEP_INDEX_URL} \
        "amd-hipcim>=26.6.0" \
    # Extract rocm-sdk-devel (thrust headers etc.) and create missing unversioned
    # .so symlinks so the linker can find libamdhip64.so etc. at build time.
    && rocm-sdk init \
    && ROCM_LIB=$(python3 -c "import _rocm_sdk_core, os; print(os.path.join(os.path.dirname(_rocm_sdk_core.__file__), 'lib'))") \
    && for lib in "${ROCM_LIB}"/lib*.so.*; do \
           [ -e "${lib}" ] || continue; \
           base=$(basename "${lib}" | sed 's/\.so\..*/\.so/'); \
           [ -e "${ROCM_LIB}/${base}" ] || ln -s "${lib}" "${ROCM_LIB}/${base}"; \
       done

# Expose rocm-sdk-devel headers (thrust, hip, etc.) to the compiler.
ENV CPATH="/opt/monai-venv/lib/python3.12/site-packages/_rocm_sdk_devel/include"

# Expose ROCm libs at runtime for cucim, torch, etc.
ENV LD_LIBRARY_PATH="/opt/monai-venv/lib/python3.12/site-packages/_rocm_sdk_core/lib"

# Verify hipCIM provides the real cucim, not the PyPI stub.
RUN python3 -c "import cucim; print('✓ hipCIM (cucim) import OK:', cucim.__version__)" \
    || { echo "✗ ERROR: amd-hipcim resolved to the public stub"; exit 1; }

RUN pip install --no-cache-dir \
    --extra-index-url ${PYTORCH_INDEX_URL} \
    --extra-index-url ${DEP_INDEX_URL} \
    -r requirements-dev.txt -c amd-constraints.txt --build-constraint amd-constraints.txt

RUN BUILD_MONAI=1 FORCE_CUDA=1 python3 setup.py bdist_wheel \
    && pip install --no-cache-dir dist/amd_monai-*.whl

RUN python3 -c "import torch; print('✓ torch:', torch.__version__)" \
    && python3 -c "import monai; print('✓ monai:', monai.__version__)"

# OMP_NUM_THREADS=1 prevents OpenBLAS thread exhaustion (core dump) in tests.
# MONAI_TEST_RERUN_FAILURES retries intermittent ROCm kernel-load failures.
ENV OMP_NUM_THREADS=1
ENV MONAI_TEST_RERUN_FAILURES=2

CMD ["bash", "runtests.sh", "--unittests"]
