.. SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

.. meta::
   :description: Overview of MONAI on ROCm, the AMD ROCm port of MONAI, including its AMD-specific optimizations and package information.
   :keywords: ROCm-LS, life sciences, MONAI overview, MONAI on ROCm, AMD ROCm port

.. _monai-overview:

**********************
MONAI on ROCm overview  
**********************

MONAI on ROCm is AMD's ROCm port of `MONAI 1.6.0 <https://monai.io>`_, a PyTorch-based framework for medical imaging tasks. The ``amd-monai`` package is API-compatible with ``monai`` and adds inference optimizations for AMD Instinct™ MI300X, MI325X, and MI355X GPUs.

AMD ROCm port
=============

The ``amd-monai`` package is an inference-optimized port of MONAI 1.6.0 targeting the AMD ROCm software stack. It is published on the AMD PyPI index and is compatible with the upstream MONAI API. No upstream model weights or configuration files are modified.

AMD-specific optimizations
==========================

.. list-table::
  :header-rows: 1
  :widths: 22 22 56

  * - Optimization
    - Component
    - Description
  * - Fused Scaled Dot-Product Attention
    - ``SwinUNETR.WindowAttention``
    - Automatically enabled on ROCm through ``torch.nn.functional.scaled_dot_product_attention`` when HIP is detected. It replaces the default multi-head attention path with a fused kernel.
  * - Dynamic graph stabilization
    - ``SlidingWindowInferer``
    - Calls ``torch._dynamo.maybe_mark_dynamic(win_data, 0)`` on ROCm so the sliding-window batch dimension is treated as dynamic by ``torch.compile``. This prevents repeated graph recompilation across windows of differing batch sizes.
  * - GEMM-based transposed convolution
    - ``DynUNet``
    - The ``use_gemm_transpose`` parameter and ``enable_gemm_transpose()`` method replace decoder ``ConvTranspose3d`` upsampling layers with a GEMM-based equivalent optimized for AMD MI300X.
  * - CUDA package exclusion
    - Package dependencies
    - ``amd-constraints.txt`` pins all ``nvidia-cu*`` packages to version 0, which prevents CUDA runtime libraries from being installed alongside ROCm.
  * - hipCIM integration
    - WSI whole-slide imaging
    - MONAI ``WSIReader`` works with ``amd-hipcim`` as the GPU-accelerated whole-slide image backend for GPU patch extraction.


Package information
===================

.. list-table::
  :header-rows: 1
  :widths: 30 70

  * - Item
    - Value
  * - Package name
    - ``amd-monai``
  * - Upstream version
    - 1.6.0
  * - Python import
    - ``import monai``
  * - Author
    - AMD Corporation
  * - PyPI index
    - ``https://pypi.amd.com/rocm-10.0.0/simple/``
  * - Source repository
    - `AMD-Ecosystem/MONAI <https://github.com/AMD-Ecosystem/MONAI>`_
  * - Documentation
    - `https://rocm.docs.amd.com/projects/monai/en/latest/ <https://rocm.docs.amd.com/projects/monai/en/latest/>`_
