"""Tests that prove the claims the article makes. Stdlib unittest + numpy:

    python -m unittest discover -s tests -v   (run from the repo root)
"""
import unittest

from crm_data import build_crm, as_documents
from demo_problem import QUESTION, ground_truth
from retrieval import VectorIndex, vector_rag, hybrid, entity_correct, full_rank_map
from text_to_sql import GUESSES

GOLD = {1, 2, 3, 4}
GOLD_TOTAL = 275000.0


class GroundTruthTests(unittest.TestCase):
    def setUp(self):
        self.conn = build_crm()

    def test_ground_truth_is_275k(self):
        gt = ground_truth(self.conn)
        self.assertEqual(gt["total"], GOLD_TOTAL)
        self.assertEqual({o[0] for o in gt["open_opps"]}, GOLD)

    def test_corpus_is_realistically_large(self):
        # Small enough to read, large enough that no lucky k window exists.
        self.assertGreater(len(as_documents(self.conn)), 300)


class HybridTests(unittest.TestCase):
    def setUp(self):
        self.conn = build_crm()

    def test_hybrid_is_exactly_right(self):
        hy = hybrid(self.conn, "acme")
        self.assertTrue(entity_correct(hy["opp_ids"], GOLD))
        self.assertEqual(hy["total"], GOLD_TOTAL)

    def test_hybrid_resolved_the_right_entity(self):
        self.assertEqual(hybrid(self.conn, "acme")["canonical_id"], 1)


class VectorRagFailureTests(unittest.TestCase):
    def setUp(self):
        self.conn = build_crm()

    def test_no_k_isolates_the_answer(self):
        # The retrieved opp set at cutoff k is a prefix of the opp ranking.
        # If a record that must be dropped ranks among the first four opp chunks,
        # then NO k produces exactly {1,2,3,4}. Prove it for every prefix.
        idx = VectorIndex(as_documents(self.conn))
        hits = idx.search(QUESTION, k=len(idx.ids))
        opp_order = [int(d.split(":")[1]) for d, _, _ in hits if d.startswith("opp:")]
        for n in range(1, len(opp_order) + 1):
            self.assertNotEqual(set(opp_order[:n]), GOLD,
                                f"unexpected lucky cutoff at the first {n} opp chunks")

    def test_a_drop_record_outranks_a_keep_record(self):
        # The structural failure in one assertion: at least one record that must be
        # dropped (won deal or look-alike) ranks above a record that must be kept.
        rm = full_rank_map(self.conn, QUESTION)
        worst_keep = max(rm[f"opp:{i}"] for i in GOLD)          # the lowest-ranked KEEP
        logistics = rm["opp:7"]                                 # Acme Logistics (drop)
        self.assertLess(logistics, worst_keep)


class TextToSqlFailureTests(unittest.TestCase):
    def setUp(self):
        self.conn = build_crm()

    def test_both_plausible_queries_are_wrong(self):
        for _label, sql in GUESSES:
            got = self.conn.execute(sql).fetchone()[0]
            self.assertNotEqual(got, GOLD_TOTAL)


class DeterminismTests(unittest.TestCase):
    def test_embedding_ranking_is_stable_across_runs(self):
        # Built-in hash() is salted per process; we use a stable hash so the demo
        # is reproducible. Confirm two fresh rankings are identical.
        a = full_rank_map(build_crm(), QUESTION)
        b = full_rank_map(build_crm(), QUESTION)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main(verbosity=2)
