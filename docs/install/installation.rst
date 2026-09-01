.. SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

.. meta::
   :description: MONAI is a domain-optimized, open-source framework based on PyTorch, designed specifically for deep learning in healthcare imaging.
   :keywords: ROCm-LS, life sciences, MONAI on ROCm installation, Build MONAI for AMD ROCm

.. _installing-monai:

********************************
MONAI on ROCm installation
********************************

MONAI on ROCm is installed in a Docker container from :ref:`AMD PyPI <package-install>` or by :ref:`building from source <source-install>`. Building from source is intended for contributors.

MONAI on ROCm works with Python 3.12 and depends on NumPy and PyTorch for AMD ROCm with optional dependencies. It supports `AMD-Ecosystem/hipCIM <https://github.com/AMD-Ecosystem/hipCIM>`_ for accelerated image loading and processing on AMD Instinct GPUs.

System environment:

+--------------+----------------+----------------+------------------------------------------+
| ROCm version | Ubuntu version | Python version | AMD Instinct™ GPU (tested)               |
+==============+================+================+==========================================+
| 10.0.0       | 24.04          | 3.12           | MI300X, MI325X, MI355X                   |
+--------------+----------------+----------------+------------------------------------------+

Launch a Docker container
===========================

Before beginning, start a Docker container with the ROCm Ubuntu Docker image from Docker Hub:

.. code:: shell

   docker run --cap-add=SYS_PTRACE --ipc=host --privileged=true \
     --shm-size=512GB --network=host --device=/dev/kfd \
     --device=/dev/dri --group-add video -it \
     -v $HOME:$HOME --name ${LOGNAME}_monai \
     ubuntu:24.04

MONAI on ROCm is installed in the Docker container. All subsequent commands must be run from within the Docker container.

Setting up the environment
==========================

Before installing MONAI on ROCm, set up the installation environment.

1. Install required system dependencies.

   .. code:: shell

      apt-get update && \
      apt-get install -y --no-install-recommends \
        python3-venv python3-pip python3-dev \
        build-essential git cmake ninja-build yasm \
        openssh-client \
        libgomp1 libstdc++-13-dev \
        libopenslide-dev libwebp-dev libzstd-dev && \
      rm -rf /var/lib/apt/lists/*

2. Create and activate the development environment.

   .. code:: shell

      python3 -m venv /opt/venv
      source /opt/venv/bin/activate
      pip install --upgrade pip

3. Install PyTorch and amd-hipcim for ROCm.

   .. code:: shell

      pip install \
          --index-url https://stable.repo.amd.com/rocm/whl-next/ \
          "rocm[libraries,devel,device-gfx942,device-gfx950]==10.0.*" \
          "torch[device-gfx942,device-gfx950]" \
          "torchvision[device-gfx942,device-gfx950]" \
          torchaudio

      pip install amd-hipcim \
          --extra-index-url=https://pypi.amd.com/rocm-10.0.0/simple/

      rocm-sdk init

4. Set environment variables.

   .. code:: shell

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
      Set ``OMP_NUM_THREADS=1`` to suppress OpenMP warnings during setup.

.. _package-install:

Install MONAI on ROCm from AMD PyPI
===================================

From within the Docker container, use these steps to install MONAI on ROCm from AMD PyPI.

1. Install optional dependencies based on the workload.

   .. code:: shell

      pip install ITK nibabel gdown tqdm lmdb psutil pandas einops mlflow \
                  pynrrd clearml transformers pydicom fire ignite         \
                  parameterized tensorboard pytorch-ignite onnx

2. Install NumPy and MONAI on ROCm.

   .. code:: shell

      pip install "numpy<2.5,>=1.24"

      pip install --no-deps amd-monai \
          --index-url https://stable.repo.amd.com/rocm/whl-next/ \
          --extra-index-url=https://pypi.amd.com/rocm-10.0.0/simple/

.. _source-install:

Build MONAI on ROCm from source
===============================

MONAI on ROCm can also be built from source if you intend to contribute to the project. The source build must also be run from within the Docker container.

1. Clone the MONAI on ROCm repository.

   .. code:: shell

      git clone git@github.com:AMD-Ecosystem/monai.git
      cd monai

2. Set build environment variables.

   The ``devel`` extra installed with ``rocm`` during environment setup provides the development packages the build needs.

   .. code:: shell

      export ROCM_DEVEL_PATH=$(python3 -c "import _rocm_sdk_devel, os; print(os.path.dirname(_rocm_sdk_devel.__file__))")
      export LIBRARY_PATH=$ROCM_DEVEL_PATH/lib:$ROCM_PATH/lib
      export CPATH=$ROCM_DEVEL_PATH/include:/usr/lib/gcc/x86_64-linux-gnu/13/include:$CPATH
      export PYTORCH_ROCM_ARCH=$AMDGPU_TARGETS

      # Symlinks required by hipcc for GPU bitcode and unversioned .so stubs
      mkdir -p $ROCM_PATH/amdgcn
      ln -sf $ROCM_PATH/lib/llvm/amdgcn/bitcode $ROCM_PATH/amdgcn/bitcode

3. Install development dependencies and build a wheel.

   .. code:: shell

      pip install -r requirements-dev.txt -c amd-constraints.txt \
          --index-url https://stable.repo.amd.com/rocm/whl-next/ \
          --extra-index-url https://pypi.org/simple/ \
          --build-constraint amd-constraints.txt

      BUILD_MONAI=1 FORCE_CUDA=1 python3 setup.py bdist_wheel
      pip install --no-deps dist/amd_monai-*.whl

   The wheel file is generated under the ``dist`` directory.

Verify installation
===================

Verify the MONAI on ROCm installation. Run these commands from within the Docker container.

.. code:: shell

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

.. code:: shell

   pip show -v amd-monai
