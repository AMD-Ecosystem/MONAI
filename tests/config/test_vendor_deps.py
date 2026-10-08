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

import types
import unittest
from unittest.mock import patch

from monai.config import vendor_deps
from monai.config.vendor_deps import (
    active_vendor,
    apply_to_dependencies,
    apply_to_optional_dependencies,
    canonical_name,
    split_requirement,
)


def fake_plugin(substitutions=None, extras=None, extra_requirements=()):
    """A minimal plugin satisfying the contract documented in vendor_deps."""
    return types.SimpleNamespace(
        NAME="fake",
        SUBSTITUTIONS=substitutions or {},
        extras_for=lambda name: (extras or {}).get(name, []),
        extra_requirements=lambda: list(extra_requirements),
    )


class TestRequirementParsing(unittest.TestCase):
    def test_canonical_name(self):
        self.assertEqual("nvidia-ml-py", canonical_name("NVIDIA_ML.Py"))

    def test_split_plain(self):
        self.assertEqual(("numpy", [], ">=1.24,<3.0"), split_requirement("numpy>=1.24,<3.0"))

    def test_split_extras_and_marker(self):
        name, extras, rest = split_requirement("cupy-cuda13x[ctk]!=14.1.0; platform_system == 'Linux'")
        self.assertEqual("cupy-cuda13x", name)
        self.assertEqual(["ctk"], extras)
        self.assertEqual("!=14.1.0; platform_system == 'Linux'", rest)

    def test_split_direct_url(self):
        name, extras, _ = split_requirement("MetricsReloaded @ git+https://github.com/x/y@z")
        self.assertEqual("MetricsReloaded", name)
        self.assertEqual([], extras)


class TestGenericRewrites(unittest.TestCase):
    def test_substitution_drop_and_dedupe(self):
        plugin = fake_plugin({"a-cu12": "repl", "a-cu13": "repl", "gone": None})
        self.assertEqual(
            {"g": ["repl", "kept"]}, apply_to_optional_dependencies({"g": ["a-cu12", "a-cu13", "gone", "kept"]}, plugin)
        )

    def test_group_can_become_empty(self):
        self.assertEqual({"g": []}, apply_to_optional_dependencies({"g": ["gone"]}, fake_plugin({"gone": None})))

    def test_extras_and_extra_requirements(self):
        plugin = fake_plugin(extras={"torch": ["device-x"]}, extra_requirements=["runtime>=1"])
        self.assertEqual(
            ["torch[device-x]>=2.8.0", "numpy>=1.24", "runtime>=1"],
            apply_to_dependencies(["torch>=2.8.0", "numpy>=1.24"], plugin),
        )

    def test_existing_extras_preserved(self):
        plugin = fake_plugin(extras={"torch": ["device-x"]})
        self.assertEqual(
            ["torch[device-x,opt-einsum]>=2.8.0"], apply_to_dependencies(["torch[opt-einsum]>=2.8.0"], plugin)
        )

    def test_unrelated_requirements_untouched(self):
        group = ["scikit-image>=0.19.0", "MetricsReloaded @ git+https://github.com/x/y@z"]
        self.assertEqual({"all": group}, apply_to_optional_dependencies({"all": group}, fake_plugin()))


class TestVendorSelection(unittest.TestCase):
    def test_none_disables_rewriting(self):
        for value in ("none", "NONE", "cpu"):
            with patch.dict("os.environ", {"MONAI_VENDOR": value}):
                self.assertIsNone(active_vendor())

    def test_unknown_vendor_is_rejected(self):
        with patch.dict("os.environ", {"MONAI_VENDOR": "nosuchvendor"}), self.assertRaises(ValueError):
            active_vendor()

    def test_explicit_vendor_bypasses_detection(self):
        with patch.dict("os.environ", {"MONAI_VENDOR": "rocm"}):
            vendor = active_vendor()
        self.assertIsNotNone(vendor)
        self.assertEqual("rocm", vendor.name)

    def test_no_vendor_when_no_probe_matches(self):
        with patch.dict("os.environ", {}, clear=True), patch.dict(vendor_deps.REGISTRY, {}, clear=True):
            self.assertIsNone(active_vendor())

    def test_plugin_not_imported_when_probe_does_not_match(self):
        probe = vendor_deps._Registration("vendor_rocm", lambda: False)
        with (
            patch.dict("os.environ", {}, clear=True),
            patch.dict(vendor_deps.REGISTRY, {"rocm": probe}, clear=True),
            patch.object(vendor_deps, "_import_plugin", side_effect=AssertionError("plugin must not be imported")),
        ):
            self.assertIsNone(active_vendor())

    def test_detection_selects_matching_plugin(self):
        probe = vendor_deps._Registration("vendor_rocm", lambda: True)
        with patch.dict("os.environ", {}, clear=True), patch.dict(vendor_deps.REGISTRY, {"rocm": probe}, clear=True):
            vendor = active_vendor()
        self.assertEqual("rocm", vendor.name)

    def test_detected_vendor_that_cannot_load_is_fatal(self):
        # Falling back to "no vendor" here would ship another vendor's packages in this one's wheel.
        probe = vendor_deps._Registration("vendor_broken", lambda: True)
        with (
            patch.dict("os.environ", {}, clear=True),
            patch.dict(vendor_deps.REGISTRY, {"rocm": probe}, clear=True),
            patch.object(vendor_deps, "_import_plugin", side_effect=SyntaxError("invalid syntax")),
            self.assertRaises(vendor_deps.VendorPluginError),
        ):
            active_vendor()

    def test_explicit_vendor_that_cannot_load_is_fatal(self):
        with (
            patch.dict("os.environ", {"MONAI_VENDOR": "rocm"}),
            patch.object(vendor_deps, "_import_plugin", side_effect=SyntaxError("invalid syntax")),
            self.assertRaises(vendor_deps.VendorPluginError),
        ):
            active_vendor()


if __name__ == "__main__":
    unittest.main()
