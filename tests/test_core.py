"""Unit tests for chunking, CUA, labeling, metrics and splits.

They use the offline HashEncoder, so no model download is needed.
"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sfl import metrics, splits, threshold  # noqa: E402
from sfl.chunking import chunk_text, split_sentences  # noqa: E402
from sfl.cua import CUA, IMPORTANT, UNIMPORTANT, sentence_features, vote  # noqa: E402
from sfl.data import load_dataset  # noqa: E402
from sfl.encoder import HashEncoder  # noqa: E402
from sfl.labeler import FactorLabeler, apply_threshold  # noqa: E402


class TestChunking(unittest.TestCase):
    def test_splits_at_chinese_numeral_headings(self):
        t = "一、甲段。二、乙段。三、丙段。"
        self.assertEqual(chunk_text(t), ["一、甲段。", "二、乙段。", "三、丙段。"])

    def test_does_not_split_inside_a_sentence(self):
        t = "一、被告於三、四月間行竊。二、被告自首。"
        self.assertEqual(len(chunk_text(t)), 2)

    def test_no_heading_gives_one_chunk(self):
        self.assertEqual(chunk_text("沒有標題的文字。"), ["沒有標題的文字。"])

    def test_empty(self):
        self.assertEqual(chunk_text("   "), [])

    def test_split_sentences(self):
        self.assertEqual(split_sentences("甲乙丙。丁戊；己", "。；", 2), ["甲乙丙", "丁戊"])


class TestCUAParts(unittest.TestCase):
    def test_features_are_the_papers_five(self):
        f = sentence_features([3, 1, 5], 1)
        self.assertEqual(set(f), {"current", "prev", "next", "is_first", "is_last"})
        self.assertEqual((f["current"], f["prev"], f["next"]), ("1", "3", "5"))
        self.assertTrue(sentence_features([3, 1, 5], 0)["is_first"])
        self.assertTrue(sentence_features([3, 1, 5], 2)["is_last"])
        self.assertNotIn("prev", sentence_features([3, 1, 5], 0))
        self.assertNotIn("next", sentence_features([3, 1, 5], 2))

    def test_vote_is_argmax(self):
        self.assertEqual(vote([IMPORTANT, UNIMPORTANT, UNIMPORTANT]), UNIMPORTANT)
        self.assertEqual(vote([IMPORTANT, IMPORTANT, UNIMPORTANT]), IMPORTANT)
        self.assertEqual(vote([IMPORTANT, UNIMPORTANT]), IMPORTANT)     # tie -> keep
        self.assertEqual(vote([]), UNIMPORTANT)


class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ds = load_dataset(ROOT / "data" / "demo")
        cls.enc = HashEncoder()
        cls.tmp = tempfile.TemporaryDirectory()
        cls.cua = CUA(cls.enc, n_clusters=8).fit(cls.ds.texts, cls.ds.passages,
                                                  str(Path(cls.tmp.name) / "crf.crfsuite"))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_demo_has_forty_passages(self):
        self.assertEqual(len(self.ds.passages), 40)

    def test_cua_has_eight_clusters(self):
        self.assertEqual(self.cua.centers.shape[0], 8)

    def test_cua_separates_boilerplate_from_narrative(self):
        boiler = "本案經檢察官提起公訴，並由本院依通常程序審理。公訴人及被告均到庭陳述意見。"
        narrative = "甲見一輛未上鎖之腳踏車停放路旁，一時貪念將之竊取。甲將竊得之物品變賣，所得款項花用殆盡。"
        self.assertEqual(self.cua.classify(boiler), UNIMPORTANT)
        self.assertEqual(self.cua.classify(narrative), IMPORTANT)

    def test_save_and_load_roundtrip_without_pickle(self):
        d = Path(self.tmp.name) / "saved"
        self.cua.save(str(d))
        self.assertEqual({p.name for p in d.iterdir()}, {"centers.npy", "crf.crfsuite", "meta.json"})
        loaded = CUA.load(str(d), self.enc)
        for t in self.ds.texts[:10]:
            for c in chunk_text(t):
                self.assertEqual(self.cua.classify(c), loaded.classify(c))

    def test_cua_predictions_are_a_subset_of_chunking_predictions(self):
        """Paper Sec. V-C: filtering at a fixed threshold can only remove predicted
        positives, so Recall cannot increase."""
        texts = self.ds.texts
        S_chunk = FactorLabeler(self.enc, self.ds.factors, "chunk").max_similarity(texts)
        S_cua = FactorLabeler(self.enc, self.ds.factors, "cua", cua=self.cua).max_similarity(texts)
        self.assertTrue(np.all(S_cua <= S_chunk + 1e-6))   # float32 matmul rounding across batch shapes
        for tau in threshold.grid() + [0.1, 0.2, 0.3]:
            p_chunk, p_cua = apply_threshold(S_chunk, tau), apply_threshold(S_cua, tau)
            self.assertTrue(np.all(p_cua <= p_chunk))
            r_chunk = metrics.compute(self.ds.Y, p_chunk)["recall"]
            r_cua = metrics.compute(self.ds.Y, p_cua)["recall"]
            self.assertLessEqual(r_cua, r_chunk + 1e-9)

    def test_empty_unit_set_means_all_factors_absent(self):
        lab = FactorLabeler(self.enc, self.ds.factors, "chunk")
        S = lab.max_similarity([""])
        self.assertTrue(np.all(np.isneginf(S)))
        self.assertTrue(np.all(apply_threshold(S, 0.0) == 0))


class TestMetrics(unittest.TestCase):
    def test_known_values(self):
        y = np.array([[1, 0], [1, 1], [0, 0], [0, 1]])
        p = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
        m = metrics.compute(y, p)           # tp=3 fp=1 fn=1 tn=3
        self.assertAlmostEqual(m["accuracy"], 75.0)
        self.assertAlmostEqual(m["precision"], 75.0)
        self.assertAlmostEqual(m["recall"], 75.0)
        self.assertAlmostEqual(m["f1"], 75.0)
        self.assertAlmostEqual(m["hamming_loss"], 25.0)

    def test_hamming_is_100_minus_accuracy(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            y = rng.integers(0, 2, (30, 6))
            p = rng.integers(0, 2, (30, 6))
            m = metrics.compute(y, p)
            self.assertAlmostEqual(m["hamming_loss"], 100 - m["accuracy"], places=9)

    def test_no_positive_predictions(self):
        m = metrics.compute(np.array([[1, 0]]), np.array([[0, 0]]))
        self.assertEqual((m["precision"], m["recall"], m["f1"]), (0.0, 0.0, 0.0))

    def test_bootstrap_interval_contains_point_estimate(self):
        rng = np.random.default_rng(1)
        y = rng.integers(0, 2, (80, 4))
        p = np.where(rng.random((80, 4)) < 0.8, y, 1 - y)
        lo, hi = metrics.bootstrap_ci(y, p, n_boot=200)["accuracy"]
        self.assertLessEqual(lo, metrics.compute(y, p)["accuracy"])
        self.assertGreaterEqual(hi, metrics.compute(y, p)["accuracy"])


class TestSplits(unittest.TestCase):
    def test_kfold_is_disjoint_and_covers_everything(self):
        ids = [f"j{i}" for i in range(50)]
        folds = splits.kfold(ids, k=2, seed=3)
        tests = []
        for train, test in folds:
            splits.assert_disjoint(train, test)
            self.assertEqual(len(train) + len(test), 50)
            tests += test
        self.assertEqual(sorted(tests), sorted(ids))

    def test_overlap_is_detected(self):
        with self.assertRaises(AssertionError):
            splits.assert_disjoint(["a", "b"], ["b", "c"])


class TestThreshold(unittest.TestCase):
    def test_select_prefers_the_best_shared_tau(self):
        Y = np.array([[1], [1], [0], [0]])
        S = {"v": np.array([[0.9], [0.8], [0.3], [0.2]])}
        self.assertEqual(threshold.select_shared(S, Y, [0.5, 0.85]), 0.5)

    def test_sweep_covers_grid(self):
        S = np.random.default_rng(0).random((10, 3))
        Y = (S > 0.5).astype(int)
        self.assertEqual([r["tau"] for r in threshold.sweep(S, Y)], threshold.grid())


if __name__ == "__main__":
    unittest.main()
