"""The numbers in configs/paper.yaml must equal the numbers stated in the paper.

PAPER below is a hand-copied list of the paper's own statements. If the paper
and the config ever disagree, this test fails.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sfl.config import paper_config  # noqa: E402
from sfl.threshold import grid  # noqa: E402


class TestPaperConsistency(unittest.TestCase):
    def setUp(self):
        self.c = paper_config()

    def test_cua(self):
        self.assertEqual(self.c["cua"]["n_clusters"], 8)                 # IV-B
        self.assertEqual(self.c["cua"]["n_labeled_passages"], 40)        # IV-B
        self.assertEqual(self.c["cua"]["crf_features"],
                         ["current", "prev", "next", "is_first", "is_last"])

    def test_finetune(self):
        f = self.c["finetune"]                                            # V-B
        self.assertEqual(f["base_model"], "BAAI/bge-m3")
        self.assertEqual(f["batch_size"], 32)
        self.assertEqual(f["group_size"], 16)
        self.assertEqual(f["epochs"], 10)
        self.assertEqual(f["learning_rate"], 2e-5)
        self.assertEqual(f["warmup_ratio"], 0.1)
        self.assertEqual(f["temperature"], 0.02)
        self.assertEqual(f["pooling"], "cls")
        self.assertEqual(f["pos_neg_ratio"], [1, 15])
        self.assertEqual(1 + f["pos_neg_ratio"][1], f["group_size"])

    def test_threshold_grid(self):
        self.assertEqual(grid(), [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85])


if __name__ == "__main__":
    unittest.main()
