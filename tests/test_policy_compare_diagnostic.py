from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

try:
    from nn.native_env import SplendorNativeEnv

    _ENV_AVAILABLE = True
except Exception:
    SplendorNativeEnv = None
    _ENV_AVAILABLE = False

if torch is not None:
    from nn.checkpoints import save_checkpoint
    from nn.model import MaskedPolicyValueNet
else:
    save_checkpoint = None
    MaskedPolicyValueNet = None


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "policy_compare_diagnostic.py"
_SPEC = importlib.util.spec_from_file_location("policy_compare_diagnostic", SCRIPT_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Failed to load script module: {SCRIPT_PATH}")
mod = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = mod
_SPEC.loader.exec_module(mod)


class TestPolicyCompareHelpers(unittest.TestCase):
    def test_masked_softmax_probs_respects_legality(self):
        logits = np.full((2, 69), -3.0, dtype=np.float64)
        mask = np.zeros((2, 69), dtype=np.bool_)
        logits[0, 0] = 1.0
        logits[0, 1] = 2.0
        logits[0, 2] = -1.0
        mask[0, 0] = True
        mask[0, 1] = True
        logits[1, 0] = 4.0
        logits[1, 1] = 0.0
        logits[1, 2] = 1.0
        mask[1, 1] = True
        mask[1, 2] = True
        probs = mod._masked_softmax_probs(logits, mask)

        self.assertEqual(probs.shape, logits.shape)
        self.assertEqual(float(probs[0, 2]), 0.0)
        self.assertEqual(float(probs[1, 0]), 0.0)
        self.assertTrue(np.allclose(np.sum(probs, axis=1), 1.0, atol=1e-6))

    def test_top1_tie_break_prefers_lowest_index(self):
        probs = np.array([[0.4, 0.4, 0.2], [0.2, 0.7, 0.7]], dtype=np.float64)
        mask = np.array([[True, True, True], [False, True, True]], dtype=np.bool_)
        idx = mod._top1_indices(probs, mask)
        self.assertEqual(int(idx[0]), 0)
        self.assertEqual(int(idx[1]), 1)

    def test_js_identical_distribution_is_zero(self):
        p = np.array([[0.5, 0.5], [0.1, 0.9]], dtype=np.float64)
        js = mod._js_divergence_per_state(p, p)
        self.assertTrue(np.allclose(js, 0.0, atol=1e-12))


@unittest.skipIf(torch is None, "torch not installed")
@unittest.skipIf(not _ENV_AVAILABLE, "native environment backend not available")
class TestPolicyCompareScript(unittest.TestCase):
    def _make_checkpoint(self, root: Path, *, run_id: str, cycle_idx: int, seed: int, policy_bias_scale: float) -> Path:
        torch.manual_seed(seed)
        model = MaskedPolicyValueNet()
        with torch.no_grad():
            model.policy_head.bias[:] = torch.linspace(
                0.0,
                policy_bias_scale,
                steps=int(model.policy_head.out_features),
                dtype=torch.float32,
            )

        info = save_checkpoint(
            model,
            output_dir=root,
            run_id=run_id,
            cycle_idx=cycle_idx,
            metadata={"seed": seed, "collector_policy": "random", "mcts_sims": 0},
        )
        return info.path

    def test_missing_checkpoint_raises(self):
        with self.assertRaises(FileNotFoundError):
            mod.main(
                [
                    "--model-a",
                    "/tmp/definitely_missing_a.pt",
                    "--model-b",
                    "/tmp/definitely_missing_b.pt",
                ]
            )

    def test_smoke_and_determinism(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            ckpt_dir = tmp / "checkpoints"
            out1 = tmp / "out1.json"
            out2 = tmp / "out2.json"

            model_a = self._make_checkpoint(
                ckpt_dir,
                run_id="testcmp",
                cycle_idx=1,
                seed=11,
                policy_bias_scale=0.10,
            )
            model_b = self._make_checkpoint(
                ckpt_dir,
                run_id="testcmp",
                cycle_idx=2,
                seed=12,
                policy_bias_scale=0.30,
            )

            args_common = [
                "--model-a",
                str(model_a),
                "--model-b",
                str(model_b),
                "--games",
                "1",
                "--max-turns",
                "6",
                "--seed",
                "123",
                "--batch-size",
                "64",
                "--mcts-sims",
                "4",
                "--mcts-eval-batch-size",
                "8",
                "--device",
                "cpu",
            ]

            rc1 = mod.main(args_common + ["--out-json", str(out1)])
            rc2 = mod.main(args_common + ["--out-json", str(out2)])
            self.assertEqual(rc1, 0)
            self.assertEqual(rc2, 0)

            payload1 = json.loads(out1.read_text(encoding="utf-8"))
            payload2 = json.loads(out2.read_text(encoding="utf-8"))

            self.assertIn("raw_head", payload1)
            self.assertIn("mcts", payload1)
            self.assertGreater(payload1["sample"]["states"], 0)
            self.assertEqual(
                payload1["raw_head"]["aggregate"],
                payload2["raw_head"]["aggregate"],
            )
            self.assertEqual(
                payload1["mcts"]["aggregate"],
                payload2["mcts"]["aggregate"],
            )


if __name__ == "__main__":
    unittest.main()
