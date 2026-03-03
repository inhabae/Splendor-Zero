#!/usr/bin/env python3
import unittest

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

if np is not None and torch is not None:
    from nn.model import MaskedPolicyValueNet
    from nn.state_schema import ACTION_DIM, STATE_DIM
    from nn.train import (
        _recommend_mcts_collector_workers,
        _model_sample_legal_action,
        masked_cross_entropy_loss,
        masked_soft_cross_entropy_loss,
        masked_logits,
        select_masked_argmax,
        select_masked_sample,
        train_one_step,
    )
else:
    MaskedPolicyValueNet = None
    ACTION_DIM = 69
    STATE_DIM = 246
    _model_sample_legal_action = None
    _recommend_mcts_collector_workers = None
    masked_cross_entropy_loss = None
    masked_soft_cross_entropy_loss = None
    masked_logits = None
    select_masked_argmax = None
    select_masked_sample = None
    train_one_step = None


@unittest.skipIf(np is None, "numpy not installed")
@unittest.skipIf(torch is None, "torch not installed")
class TestNNTrainUtils(unittest.TestCase):
    def test_recommend_mcts_collector_workers_prefers_single_for_tiny_workload(self):
        workers = _recommend_mcts_collector_workers(
            requested_workers=8,
            episodes_per_cycle=8,
            max_turns=2,
            mcts_sims=1,
        )
        self.assertEqual(workers, 1)

    def test_recommend_mcts_collector_workers_caps_by_workload(self):
        workers = _recommend_mcts_collector_workers(
            requested_workers=16,
            episodes_per_cycle=6,
            max_turns=80,
            mcts_sims=128,
        )
        self.assertEqual(workers, 3)

    def test_masked_logits_shape_mismatch_raises(self):
        logits = torch.zeros((2, ACTION_DIM))
        mask = torch.zeros((3, ACTION_DIM), dtype=torch.bool)
        with self.assertRaises(ValueError):
            masked_logits(logits, mask)

    def test_masked_logits_wrong_rank_raises(self):
        logits = torch.zeros((ACTION_DIM,))
        mask = torch.zeros((ACTION_DIM,), dtype=torch.bool)
        with self.assertRaises(ValueError):
            masked_logits(logits, mask)

    def test_masked_logits_wrong_mask_dtype_raises(self):
        logits = torch.zeros((1, ACTION_DIM))
        mask = torch.zeros((1, ACTION_DIM), dtype=torch.float32)
        with self.assertRaises(ValueError):
            masked_logits(logits, mask)

    def test_masked_logits_empty_legal_row_raises(self):
        logits = torch.zeros((1, ACTION_DIM))
        mask = torch.zeros((1, ACTION_DIM), dtype=torch.bool)
        with self.assertRaises(ValueError):
            masked_logits(logits, mask)

    def test_masked_cross_entropy_illegal_target_raises(self):
        logits = torch.zeros((1, ACTION_DIM))
        mask = torch.zeros((1, ACTION_DIM), dtype=torch.bool)
        mask[0, 3] = True
        target = torch.tensor([4], dtype=torch.long)
        with self.assertRaises(ValueError):
            masked_cross_entropy_loss(logits, mask, target)

    def test_masked_cross_entropy_target_batch_mismatch_raises(self):
        logits = torch.zeros((2, ACTION_DIM))
        mask = torch.zeros((2, ACTION_DIM), dtype=torch.bool)
        mask[:, 0] = True
        target = torch.tensor([0], dtype=torch.long)
        with self.assertRaises(ValueError):
            masked_cross_entropy_loss(logits, mask, target)

    def test_masked_soft_cross_entropy_rejects_illegal_mass(self):
        logits = torch.zeros((1, ACTION_DIM))
        mask = torch.zeros((1, ACTION_DIM), dtype=torch.bool)
        mask[0, 3] = True
        target_probs = torch.zeros((1, ACTION_DIM))
        target_probs[0, 4] = 1.0
        with self.assertRaises(ValueError):
            masked_soft_cross_entropy_loss(logits, mask, target_probs)

    def test_masked_soft_cross_entropy_rejects_non_normalized(self):
        logits = torch.zeros((1, ACTION_DIM))
        mask = torch.zeros((1, ACTION_DIM), dtype=torch.bool)
        mask[0, [3, 4]] = True
        target_probs = torch.zeros((1, ACTION_DIM))
        target_probs[0, 3] = 0.2
        target_probs[0, 4] = 0.2
        with self.assertRaises(ValueError):
            masked_soft_cross_entropy_loss(logits, mask, target_probs)

    def test_masked_soft_cross_entropy_accepts_valid_distribution(self):
        logits = torch.zeros((1, ACTION_DIM))
        mask = torch.zeros((1, ACTION_DIM), dtype=torch.bool)
        mask[0, [3, 4]] = True
        target_probs = torch.zeros((1, ACTION_DIM))
        target_probs[0, 3] = 0.25
        target_probs[0, 4] = 0.75
        loss = masked_soft_cross_entropy_loss(logits, mask, target_probs)
        self.assertTrue(torch.isfinite(loss))

    def test_select_masked_argmax_respects_legality(self):
        logits = torch.full((1, ACTION_DIM), -10.0)
        mask = torch.zeros((1, ACTION_DIM), dtype=torch.bool)
        legal = [3, 12]
        mask[0, legal] = True
        logits[0, 50] = 100.0  # illegal and highest pre-mask
        logits[0, 12] = 1.0
        idx = int(select_masked_argmax(logits, mask).item())
        self.assertIn(idx, legal)
        self.assertEqual(idx, 12)

    def test_select_masked_sample_respects_legality(self):
        logits = torch.zeros((1, ACTION_DIM))
        mask = torch.zeros((1, ACTION_DIM), dtype=torch.bool)
        legal = [3, 12, 31, 60]
        mask[0, legal] = True
        for _ in range(50):
            idx = int(select_masked_sample(logits, mask).item())
            self.assertIn(idx, legal)

    def test_model_sample_legal_action_validation_errors(self):
        model = MaskedPolicyValueNet()
        state = np.zeros((STATE_DIM,), dtype=np.float32)
        mask = np.zeros((ACTION_DIM,), dtype=np.bool_)
        mask[3] = True

        with self.assertRaises(ValueError):
            _model_sample_legal_action(model, np.zeros((STATE_DIM - 1,), dtype=np.float32), mask, device="cpu")
        with self.assertRaises(ValueError):
            _model_sample_legal_action(model, state, np.zeros((ACTION_DIM - 1,), dtype=np.bool_), device="cpu")
        with self.assertRaises(ValueError):
            _model_sample_legal_action(model, state, np.zeros((ACTION_DIM,), dtype=np.bool_), device="cpu")

    def test_train_one_step_includes_human_metrics(self):
        model = MaskedPolicyValueNet()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        batch_size = 4
        states = torch.zeros((batch_size, STATE_DIM), dtype=torch.float32)
        masks = torch.zeros((batch_size, ACTION_DIM), dtype=torch.bool)
        actions = torch.tensor([1, 2, 3, 4], dtype=torch.long)
        for i, a in enumerate(actions.tolist()):
            masks[i, a] = True
        policy = torch.zeros((batch_size, ACTION_DIM), dtype=torch.float32)
        for i, a in enumerate(actions.tolist()):
            policy[i, a] = 1.0
        value_target = torch.tensor([1.0, -1.0, 0.0, 1.0], dtype=torch.float32)
        batch = {
            "state": states,
            "mask": masks,
            "action_target": actions,
            "policy_target": policy,
            "value_target": value_target,
        }
        metrics = train_one_step(model, optimizer, batch)
        self.assertIn("action_top1_acc", metrics)
        self.assertIn("value_sign_acc", metrics)
        self.assertIn("value_mae", metrics)
        self.assertTrue(np.isfinite(metrics["action_top1_acc"]))
        self.assertTrue(np.isfinite(metrics["value_sign_acc"]))
        self.assertTrue(np.isfinite(metrics["value_mae"]))


if __name__ == "__main__":
    unittest.main()
