.. SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

.. meta::
   :description: What MONAI on ROCm is, how it relates to upstream MONAI and PyTorch for AMD ROCm, and its main features.
   :keywords: ROCm-LS, life sciences, MONAI on ROCm, What is MONAI, AMD MONAI

.. _what-is-monai:

****************
What is MONAI?
****************

MONAI on ROCm is AMD's ROCm port of `MONAI 1.6.0 <https://monai.io>`_, a PyTorch-based framework for medical imaging tasks. The ``amd-monai`` package is API-compatible with ``monai`` and adds inference optimizations for AMD Instinct™ MI300X, MI325X, and MI355X GPUs.

MONAI provides domain-optimized primitives for medical imaging deep learning workflows, including:

- Medical-image transforms and augmentation pipelines.

- Domain-specific neural network architectures, including SwinUNETR, DynUNet, and SegResNet.

- Sliding-window inference and patch-based prediction.

- Metrics, losses, and evaluation utilities for medical imaging.

- Integration with the MONAI Model Zoo for pre-trained bundle deployment.

The ``amd-monai`` package is an inference-optimized port targeting the AMD ROCm software stack. It is published on the AMD PyPI index and is compatible with the upstream MONAI API. No upstream model weights or configuration files are modified.
