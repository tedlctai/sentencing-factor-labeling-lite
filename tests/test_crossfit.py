"""Tests of the 2-fold pooled report and the paired comparison."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sfl import metrics, splits, threshold  # noqa: E402
from sfl.experiment import crossfit_report, markdown_table  # noqa: E402
from sfl.labeler import apply_threshold  # noqa: E402


def _toy(n=40, m=3, seed=0):
    rng = np.random.default_rng(seed)
    Y = (rng.random((n, m)) < 0.4).astype(int)
    base = 0.3 + 0.4 * Y + 0.1 * rng.standard_normal((n, m))
    S = {"w/o chunking": base, "w/ Chunking": base + 0.02, "w/ CUA": base - 0.02}
    ids = [f"j{i}" for i in range(n)]
    return S, Y, ids


class TestCrossfit(unittest.TestCase):
    def test_every_judgment_labeled_exactly_once_by_other_fold_tau(self):
        S, Y, ids = _toy()
        taus = [0.4, 0.5, 0.6]
        rows, preds, used = crossfit_report(S, Y, ids, taus, k=2, seed=1)
        self.assertEqual(len(used), 2)
        pos = {i: n for n, i in enumerate(ids)}
        for train_ids, test_ids in splits.kfold(ids, k=2, seed=1):
            splits.assert_disjoint(train_ids, test_ids)
            tr = [pos[i] for i in train_ids]
            te = [pos[i] for i in test_ids]
            tau = threshold.select_shared({v: m[tr] for v, m in S.items()}, Y[tr], taus)
            for v, m in S.items():
                np.testing.assert_array_equal(preds[v][te], apply_threshold(m[te], tau))
        for v in S:
            self.assertEqual(preds[v].shape, Y.shape)
        self.assertIn("/", rows[0]["tau"])
        self.assertIn("0.", markdown_table(rows))

    def test_paired_diff_of_identical_predictions_is_zero(self):
        S, Y, _ = _toy()
        p = apply_threshold(S["w/ CUA"], 0.5)
        d = metrics.paired_bootstrap_diff(Y, p, p, n_boot=100)
        for k, (diff, lo, hi) in d.items():
            self.assertEqual((diff, lo, hi), (0.0, 0.0, 0.0), k)

    def test_paired_diff_sign_and_interval_order(self):
        S, Y, _ = _toy()
        good = (Y == 1).astype(int)
        bad = np.zeros_like(good)
        d = metrics.paired_bootstrap_diff(Y, good, bad, n_boot=200)
        diff, lo, hi = d["accuracy"]
        self.assertGreater(diff, 0)
        self.assertLessEqual(lo, hi)


if __name__ == "__main__":
    unittest.main()
