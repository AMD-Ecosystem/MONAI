# License

The `amd-monai` package is an AMD port of [Project MONAI](https://github.com/Project-MONAI/MONAI). The package uses the Apache License 2.0 and preserves upstream copyright headers and license terms.

```{include} ../LICENSE
```

## Third-party dependencies

| Package | License | Use |
| --- | --- | --- |
| PyTorch | BSD 3-Clause | ROCm-enabled tensor and neural network runtime |
| NumPy | BSD 3-Clause | Array operations |
| nibabel | MIT | NIfTI and MINC input and output |
| SimpleITK | Apache 2.0 | Image input, output, and registration |
| ITK | Apache 2.0 | Image input and output |
| scikit-image | BSD 3-Clause | Image processing utilities |
| amd-hipcim | Apache 2.0 | GPU whole-slide image input and output |
| amd-cupy | MIT | Optional CuPy-compatible GPU arrays for ROCm |
| Pillow | HPND | Image format support |
| einops | MIT | Tensor rearrangement |

The MONAI on ROCm source is in [AMD-Ecosystem/MONAI](https://github.com/AMD-Ecosystem/MONAI). AMD-specific changes include fused attention, GEMM-based transposed convolution, and sliding-window graph stabilization.
