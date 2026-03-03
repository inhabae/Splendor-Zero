#!/usr/bin/env python3
import types
import unittest

from nn.native_env import _native_optimization_warning


class TestNativeEnvRuntime(unittest.TestCase):
    def test_native_optimization_warning_when_unoptimized(self):
        fake = types.SimpleNamespace(BUILD_OPTIMIZED=False, BUILD_TYPE="Debug")
        msg = _native_optimization_warning(fake)
        self.assertIsNotNone(msg)
        self.assertIn("without optimization", str(msg))

    def test_native_optimization_warning_absent_when_optimized(self):
        fake = types.SimpleNamespace(BUILD_OPTIMIZED=True, BUILD_TYPE="Release")
        self.assertIsNone(_native_optimization_warning(fake))

    def test_native_optimization_warning_absent_when_metadata_missing(self):
        fake = types.SimpleNamespace()
        self.assertIsNone(_native_optimization_warning(fake))


if __name__ == "__main__":
    unittest.main()
