.. SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

.. meta::
   :description: MONAI is a domain-optimized, open-source framework based on PyTorch, designed specifically for deep learning in healthcare imaging.
   :keywords: ROCm-LS, life sciences, MONAI on ROCm installation, Build MONAI for AMD ROCm

.. _installing-monai:

********************************
MONAI on ROCm installation
********************************

MONAI on ROCm can be installed using the package manager or from source. Package manager installation is recommended for users who don't intend to contribute to the project.

- :ref:`Use package manager <package-install>`

- :ref:`Build from source <source-install>`

System requirements
===================

+--------------+----------------+----------------+------------------------------------------+
| ROCm version | Ubuntu version | Python version | AMD Instinct™ GPU (tested)               |
+==============+================+================+==========================================+
| 10.0.0       | 24.04          | 3.12           | MI300X, MI325X, MI350X, MI355X           |
+--------------+----------------+----------------+------------------------------------------+

MONAI on ROCm requires `PyTorch for AMD ROCm <https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html>`_, which ships with ROCm. NumPy 1.24 or later is required.

Setting up the environment
==========================

Set up the environment before installing MONAI on ROCm.

1. Optionally launch a Docker container.

   Use the ROCm Ubuntu Docker image from Docker Hub:

   .. code-block:: shell

      docker run --cap-add=SYS_PTRACE --ipc=host --privileged=true   \
      --shm-size=512GB --network=host --device=/dev/kfd        \
      --device=/dev/dri --group-add video -it                  \
      -v $HOME:$HOME  --name ${LOGNAME}_monai                  \
                              rocm/dev-ubuntu-24.04:10.0.0-complete


2. Install required system dependencies.

   .. code-block:: shell

      apt-get update                                                        &&  \
      apt-get install -y software-properties-common lsb-release gnupg wget  &&  \
      apt-key adv --fetch-keys                                                  \
                  https://apt.kitware.com/keys/kitware-archive-latest.asc &&  \
      add-apt-repository -y "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main" && \
      apt-get update && \
      apt-get install -y --no-install-recommends \
         build-essential git gcc g++ cmake \
         ninja-build yasm python3-venv \
         openssh-client \
         libopenslide-dev libwebp-dev \
         libzstd-dev && \
      rm -rf /var/lib/apt/lists/* && \
      ROCM_VERSION=$(cat /opt/rocm/.info/version) && \
      UBUNTU_CODENAME=$(lsb_release -cs) && \
      echo "Detected ROCm version: ${ROCM_VERSION}, Ubuntu codename: ${UBUNTU_CODENAME}" && \
      MAJOR=$(echo ${ROCM_VERSION} | cut -d. -f1) && \
      MINOR=$(echo ${ROCM_VERSION} | cut -d. -f2) && \
      PATCH=$(echo ${ROCM_VERSION} | cut -d. -f3) && \
      PATCH=${PATCH:-0} && \
      VERNUM=$((MAJOR * 10000 + MINOR * 100 + PATCH)) && \
      if [ "${PATCH}" = "0" ]; then ROCM_SHORT_VERSION="${MAJOR}.${MINOR}"; else ROCM_SHORT_VERSION="${MAJOR}.${MINOR}.${PATCH}"; fi && \
      if ! dpkg -s amdgpu-install >/dev/null 2>&1; then \
         rm -f /etc/apt/sources.list.d/amdgpu.list /etc/apt/sources.list.d/rocm.list && \
         AMDGPU_URL="https://repo.radeon.com/amdgpu-install/${ROCM_SHORT_VERSION}/ubuntu/${UBUNTU_CODENAME}/amdgpu-install_${ROCM_SHORT_VERSION}.${VERNUM}-1_all.deb" && \
         echo "Downloading: ${AMDGPU_URL}" && \
         wget "${AMDGPU_URL}" -O amdgpu-install.deb && \
         apt-get update && \
         DEBIAN_FRONTEND=noninteractive apt-get install -y ./amdgpu-install.deb && \
         rm -f amdgpu-install.deb; \
      else \
         echo "amdgpu-install already present, skipping install"; \
      fi && \
      apt-get update && \
      apt-get install -y --no-install-recommends amdgpu-lib && \
      apt-get install -y --no-install-recommends rocjpeg rocjpeg-dev && \
      rm -rf /var/lib/apt/lists/*

3. Create and activate the development environment.

   .. code-block:: shell

      python3 -m venv monai_dev
      source monai_dev/bin/activate
      pip install --upgrade pip

4. Install PyTorch and amd-hipcim for ROCm.

   Install ``torch``, ``torchvision``, and ``torchaudio`` from the `PyTorch for AMD ROCm installation guide <https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html>`_.
   Then install ``amd-hipcim``:

   .. code-block:: shell

      pip install amd-hipcim --extra-index-url=https://pypi.amd.com/rocm-${ROCM_SHORT_VERSION}/simple/

5. Set environment variables.

   .. code-block:: shell

      export HIP_PATH=/opt/rocm
      export PATH=$HIP_PATH/bin:$PATH
      export ROCM_PATH=/opt/rocm
      export LD_LIBRARY_PATH=$HIP_PATH/lib:$LD_LIBRARY_PATH
      export ROCM_HOME=/opt/rocm
      export AMDGPU_TARGETS=gfx942
      export HIP_VISIBLE_DEVICES=0

   .. note::

      For MI300X and MI325X, set ``AMDGPU_TARGETS=gfx942``.
      For MI350X and MI355X, set ``AMDGPU_TARGETS=gfx950``.

.. _package-install:

Install MONAI on ROCm from AMD PyPI
===================================

Use these steps to install MONAI on ROCm from AMD PyPI.

1. Install optional dependencies depending on the workload:

   .. code-block:: shell

      pip install ITK nibabel gdown tqdm lmdb psutil pandas einops mlflow \
                  pynrrd clearml transformers pydicom fire ignite         \
                  parameterized tensorboard pytorch-ignite onnx

2. Install MONAI on ROCm from the AMD PyPI repository.

   .. code-block:: shell

      pip install amd-monai --extra-index-url=https://pypi.amd.com/rocm-${ROCM_SHORT_VERSION}/simple/

   .. note::

      ``amd-monai`` can pull in a CUDA build of PyTorch as a transitive dependency, replacing the ROCm build.
      If that happens, reinstall ``torch``, ``torchvision``, and ``torchaudio`` from the `PyTorch for AMD ROCm installation guide <https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html>`_.

.. _source-install:

Build MONAI on ROCm from source
===============================

MONAI on ROCm can also be built from source if you intend to contribute to the project.

1. Clone the MONAI on ROCm repository.

   .. code-block:: shell

      git clone git@github.com:ROCm-LS/monai.git
      cd monai

2. Install the development environment.

   .. code-block:: shell

      pip install -r requirements-dev.txt -c amd-constraints.txt --build-constraint amd-constraints.txt

3. Build and install with C++ extensions.

   .. code-block:: shell

      BUILD_MONAI=1 FORCE_CUDA=1 python3 setup.py develop

   To build and package an optimized wheel:

   .. code-block:: shell

      BUILD_MONAI=1 FORCE_CUDA=1 python3 setup.py develop -O1 bdist_wheel

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

The output lists ``Name: amd-monai``, ``Version: 1.6.0``, and the documentation URL ``https://rocm.docs.amd.com/projects/monai/en/latest/``.
