.. SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

.. meta::
   :description: MONAI is a domain-optimized, open-source framework based on PyTorch, designed specifically for deep learning in healthcare imaging.
   :keywords: ROCm-LS, life sciences, MONAI on ROCm installation, Build MONAI for AMD ROCm

.. _installing-monai:

===========================
MONAI on ROCm installation
===========================

To install MONAI on ROCm, you have the following options:

- :ref:`Use package manager <package-install>` (recommended)

- :ref:`Build from source <source-install>`

System requirements
--------------------

- Ubuntu version: 24.04

- ROCm version: 7.14.0

- Python version: 3.12

- AMD Instinct™ GPU: MI355X, MI325X, or MI300X

- `PyTorch for AMD ROCm <https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html>`_ version: 2.8.0 and later

- NumPy version: No earlier than 1.24 and no later than 2.4

For the complete list of dependencies, see the `requirements.txt <https://github.com/AMD-Ecosystem/MONAI/blob/main/requirements.txt>`_ file.

Setting up the environment
---------------------------

To set up the environment for MONAI on ROCm installation, follow these steps:

1. Start the ROCm PyTorch Docker container. This image ships ROCm 7.14.0, PyTorch,
   and Python 3.12, so PyTorch is already installed and no separate PyTorch install
   step is required.

   .. code-block:: shell

      docker run --cap-add=SYS_PTRACE --ipc=host --privileged=true                 \
      --shm-size=512GB --network=host --device=/dev/kfd                            \
      --device=/dev/dri --group-add video -it                                      \
      -v $HOME:$HOME  --name ${LOGNAME}_monai                                       \
      rocm/pytorch:rocm7.14_ubuntu24.04_py3.12_pytorch_release_2.12.0

   To install PyTorch for AMD ROCm on a different base image or on bare metal,
   follow the official guide at
   `PyTorch for AMD ROCm <https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html>`_.
   To install ROCm itself on a non-container host, see
   `ROCm installation <https://rocm.docs.amd.com/en/latest/install/rocm.html>`_.

2. Install the required system dependencies. ROCm (including ``rocjpeg``) is
   provided by the base image's ROCm SDK, so only the imaging build libraries are
   added here. ``build-essential`` is included because the Triton AMD backend
   JIT-compiles HIP kernels at runtime.

   .. code-block:: shell

      apt-get update && \
      apt-get install -y --no-install-recommends \
         build-essential git cmake ninja-build yasm \
         openssh-client \
         libopenslide-dev libwebp-dev libzstd-dev && \
      rm -rf /var/lib/apt/lists/*

3. Install hipCIM for accelerated digital-pathology image I/O. PyTorch is already
   present in the ``rocm/pytorch`` container started in step 1, so it is not reinstalled.

   .. code-block:: shell

      pip install --upgrade pip
      pip install amd-hipcim --extra-index-url=https://pypi.amd.com/rocm-10.0.0/simple/

4. Set the environment variables. ROCm ships as a pip package in this image, so
   ``ROCM_PATH`` points at the ROCm SDK inside the virtual environment rather than
   ``/opt/rocm``.

   .. code-block:: shell

      export ROCM_PATH=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_core
      export ROCM_HOME=$ROCM_PATH
      export ROCM_LIBRARIES_PATH=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_libraries
      export PATH=$ROCM_PATH/bin:$PATH
      # TheRock splits runtime libraries across the core and libraries trees (plus
      # nested subdirs); the math libraries that CuPy loads live in the latter.
      export LD_LIBRARY_PATH=$ROCM_PATH/lib:$ROCM_PATH/lib/rocm_sysdeps/lib:$ROCM_PATH/lib/llvm/lib:$ROCM_LIBRARIES_PATH/lib:$LD_LIBRARY_PATH
      export AMDGPU_TARGETS="gfx942;gfx950"
      export HIP_VISIBLE_DEVICES=0

.. _package-install:

Installing using package manager
----------------------------------

For MONAI on ROCm installation using a package manager, follow the steps given in this section.

1. Install the optional system dependencies depending on the workload.

   .. code-block:: shell

      pip install ITK nibabel gdown tqdm lmdb psutil pandas einops mlflow \
                  pynrrd clearml transformers pydicom fire ignite         \
                  parameterized tensorboard pytorch-ignite onnx

2. Install MONAI on ROCm from the AMD PyPI repository.

   .. code-block:: shell

      pip install amd-monai --extra-index-url=https://pypi.amd.com/rocm-10.0.0/simple/

.. _source-install:

Building from source
---------------------

To build MONAI on ROCm from source, follow the steps given in this section.

1. Download the latest version of MONAI on ROCm from the GitHub repository.

   .. code-block:: shell

      git clone git@github.com:AMD-Ecosystem/MONAI.git monai
      cd monai
2. Create and activate the development environment for building MONAI on ROCm.

   .. code-block:: shell

      pip install more-itertools
      pip install -r requirements-dev.txt -c amd-constraints.txt --build-constraint amd-constraints.txt

3. Install the ROCm devel SDK. The ROCm PyTorch runtime image ships only the ROCm
   runtime; the from-source HIP compile additionally needs the devel headers
   (rocThrust, rocPRIM, hipCUB) and the unversioned ``lib*.so`` link symlinks.

   .. code-block:: shell

      pip install "rocm-sdk-devel==7.14.0" --extra-index-url=https://repo.amd.com/rocm/whl-multi-arch/
      rocm-sdk init
      export ROCM_DEVEL_PATH=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel
      export PYTORCH_ROCM_ARCH=$AMDGPU_TARGETS
      export CPATH=$ROCM_DEVEL_PATH/include:/usr/lib/gcc/x86_64-linux-gnu/13/include:$CPATH
      export LIBRARY_PATH=$ROCM_DEVEL_PATH/lib:$ROCM_PATH/lib
      # clang expects device bitcode under $ROCM_PATH/amdgcn/bitcode
      mkdir -p $ROCM_PATH/amdgcn
      ln -sf $ROCM_PATH/lib/llvm/amdgcn/bitcode $ROCM_PATH/amdgcn/bitcode

4. Build and install MONAI on ROCm. Use ``--no-build-isolation`` so the build
   reuses the pre-installed ROCm PyTorch; build isolation would pull a CUDA build
   of PyTorch and fail.

   .. code-block:: shell

      BUILD_MONAI=1 FORCE_CUDA=1 pip install --no-build-isolation -e .

   The repository ``Dockerfile`` performs this same sequence end to end and is the
   reference for a reproducible from-source build.

Verify installation
--------------------

Use these commands to verify the MONAI on ROCm installation:

- Print the MONAI on ROCm version.

  .. code-block:: shell

   $ python -c "import monai; print(monai.__version__)"

   1.6.0

- Print the MONAI on ROCm package info.

  .. code-block:: shell

   $ pip show -v amd-monai

   Name: amd-monai
   Version: 1.6.0
   Summary: AI Toolkit for Healthcare Imaging
   Home-page: https://rocm.docs.amd.com/projects/monai/en/latest/
   Author: AMD Corporation
   Author-email:
   License: Apache License 2.0
   Location: /scratch/users/souchatt/docker/souchatt_monai/monai
   Editable project location: /scratch/users/souchatt/docker/souchatt_monai/monai
   Requires: numpy, torch
   Required-by:
   Metadata-Version: 2.1
   Installer:
   Classifiers:
      Intended Audience :: Developers
      Intended Audience :: Education
      Intended Audience :: Science/Research
      Intended Audience :: Healthcare Industry
      Programming Language :: C++
      Programming Language :: Python :: 3
      Programming Language :: Python :: 3.12
      Topic :: Scientific/Engineering
      Topic :: Scientific/Engineering :: Artificial Intelligence
      Topic :: Scientific/Engineering :: Medical Science Apps.
      Topic :: Scientific/Engineering :: Information Analysis
      Topic :: Software Development
      Topic :: Software Development :: Libraries
      Typing :: Typed
   Entry-points:
   Project-URLs:
      Documentation, https://rocm.docs.amd.com/projects/monai/en/latest/
      Bug Tracker, https://github.com/AMD-Ecosystem/MONAI/issues
      Source Code, https://github.com/AMD-Ecosystem/MONAI/
