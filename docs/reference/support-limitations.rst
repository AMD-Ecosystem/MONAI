.. SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

.. meta::
   :description: Supported features, limitations, and architecture matrix for MONAI 1.6.0 on ROCm.
   :keywords: ROCm-LS, life sciences, MONAI on ROCm features, MONAI on ROCm limitations, MONAI on ROCm known issues

.. _monai-features:

****************************************
Supported features and limitations
****************************************

The following tables list MONAI capabilities supported in the AMD ROCm release of ``amd-monai`` 1.6.0.

Supported features
==================

.. list-table::
  :header-rows: 1
  :widths: 18 28 54

  * - Feature category
    - Feature
    - Notes
  * - Inference
    - Sliding-window inference through ``SlidingWindowInferer``
    - ROCm-optimized dynamic graph stabilization prevents graph recompilation across windows when ``torch.compile`` is active.
  * - Inference
    - Patch-based and dense inference
    - Supported with GPU acceleration.
  * - Network architecture
    - SwinUNETR (3D)
    - ROCm-optimized fused scaled dot-product attention auto-enabled in ``WindowAttention`` when HIP is detected.
  * - Network architecture
    - DynUNet
    - AMD extension: ``use_gemm_transpose`` parameter and ``enable_gemm_transpose()`` method for GEMM-based transposed convolution on MI300X.
  * - Network architecture
    - VISTA3D, SegResNet, UNETR, BasicUNet, and others
    - Supported without ROCm-specific modifications.
  * - Transforms
    - GPU-accelerated transforms
    - Supported for spatial, intensity, and elastic transforms through the PyTorch HIP backend.
  * - Data loading
    - NIfTI, DICOM, MHA/MHD, PNG/JPEG
    - CPU-based I/O with GPU transfer through DataLoader.
  * - Data loading
    - Whole-slide image reading through ``WSIReader``
    - GPU-accelerated with ``amd-hipcim`` backend when ``backend="cuCIM"``.
  * - GPU acceleration
    - Mixed precision (BF16 / FP16)
    - BF16 is preferred on MI300X and MI355X through PyTorch AMP with ``torch.amp.autocast``.
  * - GPU acceleration
    - ``torch.compile`` graph optimization
    - Supported. First-call compilation latency is expected. See :ref:`monai-whats-new`.
  * - Model Zoo
    - MONAI Bundle format
    - Supported. AMD overlay mechanism adds ROCm optimizations at runtime without modifying upstream bundles.
  * - Model Zoo
    - Five validated bundles
    - VISTA3D, SwinUNETR BTCV, Whole Body CT, Spleen DeepEdit, and Pancreas DiNTS.
  * - Metrics and losses
    - DiceLoss, DiceCELoss, FocalLoss, Hausdorff distance, MeanIoU
    - Supported.
  * - Interoperability
    - NumPy, ITK, SimpleITK array conversions
    - Supported through the CPU bridge.
  * - Interoperability
    - ``amd-cupy``
    - Supported with ``amd-cupy`` 14.1.1 or later.
  * - Foundation model
    - EXAONEPath 2.0
    - Computational pathology foundation model validated on AMD hardware. ViT-based WSI patch inference through Hugging Face.

Limitations

.. list-table::
  :header-rows: 1
  :widths: 18 14 22 46

  * - Architecture
    - ROCm support
    - AMD extensions
    - Tasks
  * - SwinUNETR
    - Supported
    - Fused SDPA auto-enable
    - CT/MRI segmentation, whole-body
  * - DynUNet
    - Supported
    - GEMM-based transpose conv
    - Segmentation, nnU-Net backbone
  * - VISTA3D
    - Supported
    - BF16 and ``torch.compile`` overlay
    - Universal volumetric segmentation
  * - SegResNet
    - Supported
    - None
    - Brain tumor segmentation (BraTS)
  * - UNETR
    - Supported
    - None
    - Transformer-based segmentation
  * - BasicUNet
    - Supported
    - None
    - General encoder-decoder
  * - DiNTS (NAS)
    - Supported
    - None
    - NAS-discovered segmentation
  * - EXAONEPath 2.0
    - Supported through Hugging Face
    - None
    - Computational pathology, WSI

hipCIM integration example
==========================

The following example uses GPU-accelerated whole-slide image I/O through ``amd-hipcim``:

.. code-block:: python

   from monai.data import WSIReader

   reader = WSIReader(backend="cuCIM")
   wsi = reader.read("slide_path.svs")
   patch = reader.get_data(wsi, location=(1000, 2000), size=(256, 256), level=0)

CuPy interoperability example
=============================

The following example uses ``amd-cupy`` for GPU-to-GPU data transfers without a CPU round trip:

.. code-block:: python

   import cupy as cp
   import torch
   from monai.transforms import CuCIM

   cupy_array = cp.random.rand(1, 128, 128, 64).astype(cp.float32)
   torch_tensor = torch.as_tensor(cupy_array, device="cuda")
   print(torch_tensor.device)
