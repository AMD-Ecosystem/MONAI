# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import unittest

import torch
from parameterized import parameterized

from monai.networks import eval_mode
from monai.networks.blocks.dynunet_block import UnetBasicBlock, UnetResBlock, UnetUpBlock, get_padding
from tests.test_utils import assert_allclose, dict_product, test_script_save

TEST_CASE_RES_BASIC_BLOCK = []
for params in dict_product(
    spatial_dims=range(2, 4),
    kernel_size=[1, 3],
    stride=[1, 2],
    norm_name=[("GROUP", {"num_groups": 16}), ("batch", {"track_running_stats": False}), "instance"],
    in_size=[15, 16],
):
    padding = get_padding(params["kernel_size"], params["stride"])
    if not isinstance(padding, int):
        padding = padding[0]
    out_size = int((params["in_size"] + 2 * padding - params["kernel_size"]) / params["stride"]) + 1
    test_case = [
        {
            **{k: v for k, v in params.items() if k != "in_size"},
            "in_channels": 16,
            "out_channels": 16,
            "act_name": ("leakyrelu", {"inplace": True, "negative_slope": 0.1}),
        },
        (1, 16, *([params["in_size"]] * params["spatial_dims"])),
        (1, 16, *([out_size] * params["spatial_dims"])),
    ]
    TEST_CASE_RES_BASIC_BLOCK.append(test_case)

TEST_UP_BLOCK = []
in_channels, out_channels = 4, 2
for params in dict_product(
    spatial_dims=range(2, 4),
    kernel_size=[1, 3],
    stride=[1, 2],
    norm_name=["batch", "instance"],
    in_size=[15, 16],
    trans_bias=[True, False],
):
    out_size = params["in_size"] * params["stride"]
    test_case = [
        {
            **{k: v for k, v in params.items() if k != "in_size"},
            "in_channels": in_channels,
            "out_channels": out_channels,
            "upsample_kernel_size": params["stride"],
        },
        (1, in_channels, *([params["in_size"]] * params["spatial_dims"])),
        (1, out_channels, *([out_size] * params["spatial_dims"])),
        (1, out_channels, *([params["in_size"] * params["stride"]] * params["spatial_dims"])),
    ]
    TEST_UP_BLOCK.append(test_case)


class TestResBasicBlock(unittest.TestCase):
    @parameterized.expand(TEST_CASE_RES_BASIC_BLOCK)
    def test_shape(self, input_param, input_shape, expected_shape):
        for net in [UnetResBlock(**input_param), UnetBasicBlock(**input_param)]:
            with eval_mode(net):
                result = net(torch.randn(input_shape))
                self.assertEqual(result.shape, expected_shape)

    def test_ill_arg(self):
        with self.assertRaises(ValueError):
            UnetBasicBlock(3, 4, 2, kernel_size=3, stride=1, norm_name="norm")
        with self.assertRaises(AssertionError):
            UnetResBlock(3, 4, 2, kernel_size=1, stride=4, norm_name="batch")

    def test_script(self):
        input_param, input_shape, _ = TEST_CASE_RES_BASIC_BLOCK[0]

        for net_type in (UnetResBlock, UnetBasicBlock):
            net = net_type(**input_param)
            test_data = torch.randn(input_shape)
            test_script_save(net, test_data)


class TestUpBlock(unittest.TestCase):
    @parameterized.expand(TEST_UP_BLOCK)
    def test_shape(self, input_param, input_shape, expected_shape, skip_shape):
        net = UnetUpBlock(**input_param)
        with eval_mode(net):
            result = net(torch.randn(input_shape), torch.randn(skip_shape))
            self.assertEqual(result.shape, expected_shape)

    def test_script(self):
        input_param, input_shape, _, skip_shape = TEST_UP_BLOCK[0]

        net = UnetUpBlock(**input_param)
        test_data = torch.randn(input_shape)
        skip_data = torch.randn(skip_shape)
        test_script_save(net, test_data, skip_data)


class TestUpBlockGemmTranspose(unittest.TestCase):
    """AMD MI GPUs: the opt-in pixel-shuffle GEMM decomposition of the decoder
    ConvTranspose3d (kernel_size == stride) must be numerically identical to the
    stock transposed convolution it replaces."""

    def test_gemm_decomposition_equivalence(self):
        # exercise the decomposition math directly so the check is meaningful on
        # any platform (the runtime gate is ROCm-only, but the math is not).
        net = UnetUpBlock(
            spatial_dims=3, in_channels=4, out_channels=2, kernel_size=3, stride=2, norm_name="instance", upsample_kernel_size=2
        )
        x = torch.randn(1, 4, 5, 6, 7)
        with eval_mode(net):
            expected = net.transp_conv(x)
            result = net._transp_conv_gemm(x)
        self.assertEqual(result.shape, expected.shape)
        assert_allclose(result, expected, atol=1e-4, rtol=1e-4)

    def test_gemm_decomposition_falls_through_when_k_ne_s(self):
        # kernel_size != stride violates the zero-overlap precondition; the
        # decomposition must fall back to the stock transposed convolution.
        net = UnetUpBlock(
            spatial_dims=3, in_channels=4, out_channels=2, kernel_size=3, stride=2, norm_name="instance", upsample_kernel_size=3
        )
        x = torch.randn(1, 4, 5, 6, 7)
        with eval_mode(net):
            expected = net.transp_conv(x)
            result = net._transp_conv_gemm(x)
        assert_allclose(result, expected, atol=1e-4, rtol=1e-4)

    def test_forward_equivalence(self):
        # two blocks sharing weights, GEMM path on vs off, must agree end-to-end.
        params = dict(
            spatial_dims=3, in_channels=4, out_channels=2, kernel_size=3, stride=2, norm_name="instance", upsample_kernel_size=2
        )
        net_gemm = UnetUpBlock(use_gemm_transpose=True, **params)
        net_ref = UnetUpBlock(use_gemm_transpose=False, **params)
        net_gemm.load_state_dict(net_ref.state_dict())
        inp = torch.randn(1, 4, 4, 4, 4)
        skip = torch.randn(1, 2, 8, 8, 8)
        with eval_mode(net_gemm), eval_mode(net_ref):
            out_gemm = net_gemm(inp, skip)
            out_ref = net_ref(inp, skip)
        assert_allclose(out_gemm, out_ref, atol=1e-4, rtol=1e-4)

    def test_gate_requires_rocm(self):
        # the runtime gate must be off on non-ROCm builds even when opted in.
        net = UnetUpBlock(
            spatial_dims=3, in_channels=4, out_channels=2, kernel_size=3, stride=2,
            norm_name="instance", upsample_kernel_size=2, use_gemm_transpose=True,
        )
        self.assertEqual(net._use_gemm_transpose, torch.version.hip is not None)


if __name__ == "__main__":
    unittest.main()
