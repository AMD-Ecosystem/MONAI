.. SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

.. meta::
   :description: What's new in MONAI 1.6.0 on ROCm, including AMD ROCm optimizations, upstream MONAI 1.6.0 highlights, and known issues.
   :keywords: ROCm-LS, life sciences, MONAI release notes, MONAI on ROCm, MONAI 1.6.0

.. _monai-whats-new:

*****************************
What's new in MONAI on ROCm
*****************************

Release information
===================

.. list-table::
  :header-rows: 1
  :widths: 40 60

  * - Field
    - Value
  * - Release date
    - 2026-08
  * - Package
    - ``amd-monai``
  * - Upstream MONAI version
    - 1.6.0
  * - ROCm version
    - 10.0.0
  * - Python version
    - 3.12
  * - Supported GPUs
    - MI300X (gfx942), MI325X (gfx942), MI350X (gfx950), MI355X (gfx950)

AMD ROCm optimizations
======================

.. list-table::
  :header-rows: 1
  :widths: 22 38 40

  * - Component
    - Change
    - Benefit
  * - ``SwinUNETR.WindowAttention``
    - Fused Scaled Dot-Product Attention auto-enabled on ROCm when ``torch.version.hip is not None``
    - Reduced attention memory overhead and improved SwinUNETR throughput on MI300-class hardware
  * - ``SlidingWindowInferer``
    - ``torch._dynamo.maybe_mark_dynamic(win_data, 0)`` applied on ROCm before each inference window
    - Eliminates repeated graph recompilation across windows with varying batch sizes when ``torch.compile`` is active
  * - ``DynUNet``
    - New ``use_gemm_transpose`` constructor parameter and ``enable_gemm_transpose()`` post-construction method
    - Replaces ``ConvTranspose3d`` decoder upsampling with a GEMM-based equivalent optimized for AMD MI300X matrix engines. No-op on non-AMD hardware.
  * - Package dependencies
    - ``amd-constraints.txt`` pins all ``nvidia-cu*`` packages to version 0
    - Prevents CUDA runtime libraries from being installed in ROCm environments and eliminates import-time conflicts
  * - WSI / hipCIM integration
    - ``WSIReader(backend="cuCIM")`` validated with ``amd-hipcim`` 26.06.00
    - GPU-accelerated whole-slide image patch extraction on ROCm
  * - MONAI Model Zoo overlays
    - AMD ROCm overlay configs for five validated bundles
    - Applies channels_last_3d memory format, BF16 AMP, ``torch.compile``, and device-aware checkpoint loading at runtime without modifying upstream bundle files
  * - EXAONEPath 2.0 on ROCm
    - LGAI Research computational pathology foundation model validated on AMD Instinct GPUs
    - Advanced pathology foundation model support on AMD hardware.

Upstream MONAI 1.6.0 highlights
=================================

``amd-monai`` 1.6.0 is built on `MONAI 1.6.0 <https://github.com/Project-MONAI/MONAI/releases/tag/1.6.0>`_, released 2026-06-12, with upstream API compatibility. Notable upstream additions include:

- ``MAPEMetric`` and ``CalibrationErrorMetric`` for regression and calibration evaluation.

- ``MCCLoss`` and ``AUCMLoss`` for classification tasks with imbalanced labels.

- A 3D ``PanopticQualityMetric`` for panoptic segmentation.

- Gradient accumulation in ``SupervisedTrainer`` through the ``accumulation_steps`` parameter for training with larger effective batch sizes on memory-constrained GPUs.

For the complete list of upstream changes, see the `MONAI 1.6.0 release notes <https://github.com/Project-MONAI/MONAI/releases/tag/1.6.0>`_.

Known issues
============

.. list-table::
  :header-rows: 1
  :widths: 28 36 36

  * - Issue
    - Details
    - Workaround
  * - ``torch.compile`` first-call latency
    - When Model Zoo ROCm overlays enable ``torch.compile``, the first inference call triggers HIP kernel compilation, adding 30 to 120 seconds depending on model size and GPU type.
    - Warm up the model with one dummy call before benchmarking. Compiled kernels are cached and reused across sessions.
  * - BF16 numerical delta vs FP32
    - BF16 AMP introduces small numerical differences, about 1e-3, compared to FP32. Dice scores and other metrics might differ slightly from FP32 reference values.
    - Use FP32 for result validation. BF16 is recommended for production throughput workloads where small numerical deltas are acceptable.
  * - GPU direct storage through KvikIO or cuFile
    - Not supported on ROCm.
    - Use standard CPU-mediated data loading. For WSI, use ``amd-hipcim`` for GPU-accelerated patch extraction.
  * - NVTX profiling markers
    - MONAI NVTX-based profiling annotations are no-ops on ROCm.
    - Use ``rocprof`` or Omniperf directly for hardware-level profiling.
  * - ``amd-cupy`` requirement
    - CuPy-based transforms and post-processing require ``amd-cupy`` 14.1.1 or later. Standard NVIDIA CuPy packages are not compatible on ROCm.
    - Install ``amd-cupy`` from AMD PyPI.
