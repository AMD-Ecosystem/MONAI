.. SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

.. meta::
   :description: MONAI is a domain-optimized, open-source framework based on PyTorch, designed specifically for deep learning in healthcare imaging.
   :keywords: ROCm-LS, life sciences, MONAI on ROCm installation, Build MONAI for AMD ROCm

.. _installing-monai:

********************************
MONAI on ROCm installation
********************************

MONAI on ROCm can be installed using :ref:`the package manager <package-install>` or by :ref:`building from source <source-install>`. Package manager installation is recommended for users who don't intend to contribute to the project.

System environment:

+--------------+----------------+----------------+------------------------------------------+
| ROCm version | Ubuntu version | Python version | AMD Instinct™ GPU (tested)               |
+==============+================+================+==========================================+
| 10.0.0       | 24.04          | 3.12           | MI300X, MI325X, MI355X                   |
+--------------+----------------+----------------+------------------------------------------+

Setting up the environment
==========================

Set up the environment before installing MONAI on ROCm.

1. Optionally launch a Docker container.

   Use the ROCm Ubuntu Docker image from Docker Hub:

   .. code-block:: shell

      docker run --cap-add=SYS_PTRACE --ipc=host --privileged=true   \
        --shm-size=512GB --network=host --device=/dev/kfd        \
        --device=/dev/dri --group-add video -it                  \
        -v $HOME:$HOME --name ${LOGNAME}_monai                  \
        ubuntu:24.04

2. Install required system dependencies.

   .. code-block:: shell

      apt-get update && \
      apt-get install -y --no-install-recommends \
        python3-venv python3-pip python3-dev \
        build-essential git cmake ninja-build yasm \
        openssh-client \
        libgomp1 libstdc++-13-dev \
        libopenslide-dev libwebp-dev libzstd-dev && \
      rm -rf /var/lib/apt/lists/*

3. Create and activate the development environment.

   .. code-block:: shell

      python3 -m venv /opt/venv
      source /opt/venv/bin/activate
      pip install --upgrade pip

4. Install PyTorch and amd-hipcim for ROCm.

   .. code-block:: shell

      pip install \
          --index-url https://rocm.nightlies.amd.com/whl-multi-arch/ \
          "rocm[libraries,devel,device-gfx942,device-gfx950]" \
          "torch[device-gfx942,device-gfx950]" \
          "torchvision[device-gfx942,device-gfx950]" \
          torchaudio

      pip install amd-hipcim \
          --extra-index-url=https://pypi.amd.com/rocm-10.0.0/simple/

      rocm-sdk init

5. Set environment variables.

   .. code-block:: shell

      export ROCM_PATH=$(python3 -c "import _rocm_sdk_core, os; print(os.path.dirname(_rocm_sdk_core.__file__))")
      export ROCM_HOME=$ROCM_PATH
      export ROCM_LIBRARIES_PATH=$(python3 -c "import _rocm_sdk_libraries, os; print(os.path.dirname(_rocm_sdk_libraries.__file__))")
      export PATH=$ROCM_PATH/bin:$PATH
      export LD_LIBRARY_PATH=$ROCM_PATH/lib:$ROCM_PATH/lib/rocm_sysdeps/lib:$ROCM_PATH/lib/llvm/lib:$ROCM_LIBRARIES_PATH/lib:$LD_LIBRARY_PATH
      export AMDGPU_TARGETS="gfx942"
      export OMP_NUM_THREADS=1

   .. note::

      For MI300X and MI325X, set ``AMDGPU_TARGETS=gfx942``.
      For MI355X, set ``AMDGPU_TARGETS=gfx950``.

.. _package-install:

Install MONAI on ROCm from AMD PyPI
===================================

Use these steps to install MONAI on ROCm from AMD PyPI.

1. Install optional dependencies depending on the workload:

   .. code-block:: shell

      pip install ITK nibabel gdown tqdm lmdb psutil pandas einops mlflow \
                  pynrrd clearml transformers pydicom fire ignite         \
                  parameterized tensorboard pytorch-ignite onnx

2. Install MONAI on ROCm from the AMD PyPI repository:

   .. code-block:: shell

      pip install amd-monai \
          --index-url https://rocm.nightlies.amd.com/whl-multi-arch/ \
          --extra-index-url=https://pypi.amd.com/rocm-10.0.0/simple/

.. _source-install:

Build MONAI on ROCm from source
===============================

MONAI on ROCm can also be built from source if you intend to contribute to the project.

1. Clone the MONAI on ROCm repository:

   .. code-block:: shell

      git clone git@github.com:AMD-Ecosystem/MONAI.git
      cd monai

2. Install ROCm development packages and set build environment variables:

   .. code-block:: shell

      pip install \
          --index-url https://rocm.nightlies.amd.com/whl-multi-arch/ \
          "rocm[devel]"

      export ROCM_DEVEL_PATH=$(python3 -c "import _rocm_sdk_devel, os; print(os.path.dirname(_rocm_sdk_devel.__file__))")
      export LIBRARY_PATH=$ROCM_DEVEL_PATH/lib:$ROCM_PATH/lib
      export CPATH=$ROCM_DEVEL_PATH/include:/usr/lib/gcc/x86_64-linux-gnu/13/include:$CPATH
      export PYTORCH_ROCM_ARCH=$AMDGPU_TARGETS

      mkdir -p $ROCM_PATH/amdgcn
      ln -sf $ROCM_PATH/lib/llvm/amdgcn/bitcode $ROCM_PATH/amdgcn/bitcode

3. Install development dependencies and build a wheel:

   .. code-block:: shell

      pip install -r requirements-dev.txt -c amd-constraints.txt \
          --index-url https://rocm.nightlies.amd.com/whl-multi-arch/ \
          --extra-index-url https://pypi.org/simple/ \
          --build-constraint amd-constraints.txt

      BUILD_MONAI=1 FORCE_CUDA=1 python3 setup.py bdist_wheel
      pip install dist/amd_monai-*.whl

   The wheel file is generated under the ``dist`` directory.

Verify installation
===================

Verify the MONAI on ROCm installation.

.. code-block:: shell

   python3 -c "import monai; print(monai.__version__)"

.. code-block:: python

   import torch
   import monai

   print(f"MONAI version: {monai.__version__}")
   print(f"PyTorch version: {torch.__version__}")
   print(f"ROCm available: {torch.version.hip is not None}")
   print(f"GPU available: {torch.cuda.is_available()}")
   if torch.cuda.is_available():
       print(f"GPU: {torch.cuda.get_device_name(0)}")

.. code-block:: shell

   pip show -v amd-monai
