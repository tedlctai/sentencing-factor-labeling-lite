"""Tests of the verifier, the fine-tuning data builder and the split (no network, no torch)."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sfl.chunking import chunk_text  # noqa: E402
from sfl.data import load_dataset, load_split  # noqa: E402
from sfl.encoder import HashEncoder  # noqa: E402
from sfl.cua import CUA  # noqa: E402
from sfl.finetune_data import (build_entries, candidate_chunks, split_entries, table1_summary,  # noqa: E402
                               verify_top_k)
from sfl.train import info_nce_mask  # noqa: E402
from sfl.verifier import LLMVerifier, PrecomputedVerifier, parse_answer  # noqa: E402

DEMO = ROOT / "data" / "demo"


class TestVerifier(unittest.TestCase):
    def test_parse(self):
        self.assertTrue(parse_answer(" Y"))
        self.assertFalse(parse_answer("N."))
        with self.assertRaises(ValueError):
            parse_answer("maybe")

    def test_llm_verifier_uses_cache(self):
        sent = []

        def fake_post(url, headers, payload, timeout):
            sent.append(payload)
            return {"choices": [{"message": {"content": "Y" if "刀" in payload["messages"][0]["content"] else "N"}}]}

        with tempfile.TemporaryDirectory() as d:
            cache = str(Path(d) / "c.jsonl")
            v = LLMVerifier("some-model", api_key="k", cache_path=cache, post=fake_post)
            self.assertTrue(v("攜帶兇器行竊", "被告持刀"))
            self.assertFalse(v("攜帶兇器行竊", "被告空手"))
            self.assertTrue(v("攜帶兇器行竊", "被告持刀"))      # from cache
            self.assertEqual(len(sent), 2)
            self.assertEqual(sent[0]["model"], "some-model")
            v2 = LLMVerifier("some-model", api_key="k", cache_path=cache, post=fake_post)   # re-run
            self.assertTrue(v2("攜帶兇器行竊", "被告持刀"))
            self.assertEqual(v2.calls, 0)

    def test_no_key_raises(self):
        v = LLMVerifier("m", api_key="", api_key_env="SFL_TEST_NO_SUCH_KEY", post=lambda *a: {})
        with self.assertRaises(RuntimeError):
            v("f", "c")

    def test_precomputed(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "v.jsonl"
            p.write_text(json.dumps({"id": "x", "factor": "f", "chunk": "c", "relevant": True}) + "\n", encoding="utf-8")
            v = PrecomputedVerifier(p)
            self.assertTrue(v("f", "c"))
            with self.assertRaises(KeyError):
                v("f", "other")


class TestFineTuneData(unittest.TestCase):
    def setUp(self):
        self.ds = load_dataset(DEMO)
        self.split = load_split(DEMO, "sft")

    def test_split_is_disjoint_and_complete(self):
        self.assertFalse(set(self.split["train"]) & set(self.split["test"]))
        self.assertEqual(set(self.split["train"]) | set(self.split["test"]), set(self.ds.ids))

    def test_overlapping_split_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "splits").mkdir()
            (Path(d) / "splits" / "bad.json").write_text(json.dumps({"train": ["a", "b"], "test": ["b"]}))
            with self.assertRaises(ValueError):
                load_split(d, "bad")

    def test_entries_come_only_from_the_training_pool(self):
        pool = self.ds.subset(self.split["train"][:20])
        verifier = lambda f, c: "水果刀" in c and f == self.ds.factors[0]
        recs = verify_top_k(HashEncoder(), pool.texts, pool.ids, pool.factors, verifier, top_k=5)
        self.assertEqual({r["id"] for r in recs}, set(pool.ids))
        chunks = {i: chunk_text(t) for i, t in zip(pool.ids, pool.texts)}
        entries = build_entries(recs, chunks, group_size=8, seed=1)
        test_texts = " ".join(self.ds.subset(self.split["test"]).texts)
        self.assertTrue(entries)
        for e in entries:
            self.assertEqual(e["query"], self.ds.factors[0])        # only that factor has positives
            self.assertEqual(len(e["neg"]), 7)
            self.assertNotIn(e["pos"][0], e["neg"])
            self.assertTrue(verifier(e["query"], e["pos"][0]))
            self.assertNotIn(e["source_id"], self.split["test"])
            for c in e["pos"] + e["neg"]:
                self.assertTrue(any(c in t for t in pool.texts))     # taken from the training pool
        # a negative never carries the factor according to the verifier
        for e in entries:
            for c in e["neg"]:
                self.assertFalse(verifier(e["query"], c))
        self.assertTrue(test_texts)

    def test_random_source_ignores_rejections_and_never_uses_positives(self):
        pool = self.ds.subset(self.split["train"][:20])
        verifier = lambda f, c: "水果刀" in c and f == self.ds.factors[0]
        recs = verify_top_k(HashEncoder(), pool.texts, pool.ids, pool.factors, verifier, top_k=2)
        chunks = {i: chunk_text(t) for i, t in zip(pool.ids, pool.texts)}
        e1 = build_entries(recs, chunks, group_size=8, seed=1)
        self.assertTrue(e1)
        for x in e1:
            self.assertEqual(len(x["neg"]), 7)
            self.assertFalse(any(verifier(x["query"], c) for c in x["neg"]))   # no accepted positive as negative
        # rejections are not a signal: dropping every rejected record leaves the entries unchanged
        f0 = self.ds.factors[0]
        only_pos = [r for r in recs if r["relevant"] or r["factor"] != f0]
        e2 = build_entries(only_pos, chunks, group_size=8, seed=1)
        key = lambda es: sorted((x["query"], x["pos"][0], tuple(x["neg"])) for x in es if x["query"] == f0)
        self.assertEqual(key(e1), key(e2))

    def test_candidates_come_from_cua_retained_chunks(self):
        """Paper IV-C step 1: the CUA filters chunks before retrieval and negatives."""
        with tempfile.TemporaryDirectory() as d:
            pool = self.ds.subset(self.split["train"])
            cua = CUA(HashEncoder(), n_clusters=8).fit(pool.texts, self.ds.passages, str(Path(d) / "crf"))
            some_filtered = False
            for t in pool.texts[:15]:
                kept = candidate_chunks(t, cua)
                self.assertTrue(set(kept) <= set(chunk_text(t)))
                some_filtered |= len(kept) < len(chunk_text(t))
            self.assertTrue(some_filtered)                       # the demo's boilerplate chunks are dropped
            verifier = lambda f, c: "水果刀" in c
            recs = verify_top_k(HashEncoder(), pool.texts[:15], pool.ids[:15], pool.factors, verifier, top_k=5, cua=cua)
            kept_all = {c for t in pool.texts[:15] for c in candidate_chunks(t, cua)}
            self.assertTrue({r["chunk"] for r in recs} <= kept_all)

    def test_entry_split_is_ninety_ten_and_summary_matches(self):
        entries = [{"query": "q", "pos": [str(i)], "neg": ["n"] * 15, "source_id": f"j{i % 7}"} for i in range(200)]
        train, dev = split_entries(entries, seed=3)
        self.assertEqual((len(train), len(dev)), (180, 20))
        self.assertFalse({e["pos"][0] for e in train} & {e["pos"][0] for e in dev})
        info = table1_summary(entries, train, dev, n_judgments=40, n_factors=5, group_size=16)
        self.assertEqual(info["pos_neg"], "1:15")
        self.assertAlmostEqual(info["train_share"], 0.9)
        self.assertAlmostEqual(info["entries_per_judgment_factor_pair"], 1.0)

    def test_entries_per_positive_multiplies_entries_with_fresh_negatives(self):
        pool = self.ds.subset(self.split["train"][:30])
        verifier = lambda f, c: "水果刀" in c and f == self.ds.factors[0]
        recs = verify_top_k(HashEncoder(), pool.texts, pool.ids, pool.factors, verifier, top_k=3)
        chunks = {i: chunk_text(t) for i, t in zip(pool.ids, pool.texts)}
        one = build_entries(recs, chunks, group_size=16, entries_per_positive=1, seed=1)
        three = build_entries(recs, chunks, group_size=16, entries_per_positive=3, seed=1)
        self.assertEqual(len(three), 3 * len(one))
        self.assertTrue(all(len(e["neg"]) == 15 for e in three))

    def test_dev_fraction_zero_keeps_every_entry(self):
        es = [{"query": "q", "pos": [str(i)], "neg": []} for i in range(7)]
        train, dev = split_entries(es, 0.0, seed=1)
        self.assertEqual((len(train), len(dev)), (7, 0))
        train, dev = split_entries(es, 0.1, seed=1)
        self.assertEqual((len(train), len(dev)), (6, 1))

    def test_same_query_mask(self):
        m = info_nce_mask(["a", "b", "a"])
        self.assertTrue(m[0][2] and m[2][0])
        self.assertFalse(m[0][0] or m[0][1] or m[1][2])


if __name__ == "__main__":
    unittest.main()


class TestDevice(unittest.TestCase):
    def test_requested_device_is_returned_unchanged(self):
        from sfl.device import pick_device
        for d in ("cuda", "mps", "cpu"):
            self.assertEqual(pick_device(d), d)

    def test_auto_choice_is_one_of_the_known_devices(self):
        from sfl.device import pick_device
        self.assertIn(pick_device(), ("cuda", "mps", "cpu"))


class TestCrossfitScript(unittest.TestCase):
    def test_load_split_by_path_and_plumbing_run(self):
        import json
        import subprocess
        import sys
        import tempfile
        from pathlib import Path

        from sfl.data import load_split

        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "s.json"
            p.write_text(json.dumps({"train": ["a"], "test": ["b"]}))
            self.assertEqual(load_split(root / "data/demo", str(p))["test"], ["b"])
            out = Path(tmp) / "cf"
            r = subprocess.run([sys.executable, str(root / "scripts/crossfit_sft.py"), "--encoder", "hash",
                                "--skip-train", "--verifier", "mock", "--folds", "2", "--out", str(out),
                                "--taus", "0.10,0.15,0.20,0.25,0.30"], cwd=root, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr[-800:])
            self.assertTrue((out / "results.md").exists())
            # every judgment appears in exactly one fold's held-out list
            seen = []
            for f in (0, 1):
                seen += json.loads((out / f"fold{f}" / "split.json").read_text())["test"]
            self.assertEqual(len(seen), len(set(seen)))
            self.assertEqual(len(seen), len(load_dataset(root / "data/demo").ids))
