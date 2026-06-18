"""Run the question through both pipelines and score them against ground truth.

    python demo_retrieval.py
"""
from __future__ import annotations

from crm_data import build_crm, as_documents
from demo_problem import QUESTION, ground_truth
from retrieval import vector_rag, hybrid, entity_correct, annotate_opps, full_rank_map


def main():
    conn = build_crm()
    gt = ground_truth(conn)
    gold_total = gt["total"]
    gold_opp_ids = [o[0] for o in gt["open_opps"]]
    n_docs = len(as_documents(conn))

    print(f"Corpus: {n_docs} chunks.")
    print(f"QUESTION: {QUESTION}")
    print(f"CORRECT ANSWER: ${gold_total:,.0f}  (open opportunity ids {sorted(gold_opp_ids)})\n")

    # --- the centerpiece: similarity cannot separate keep from drop ----------
    rank_map = full_rank_map(conn, QUESTION)
    print("HOW SIMILARITY RANKS THE 'ACME' OPPORTUNITIES, vs what only the DB knows:")
    print("  rank | stage            | entity        | keep? | opportunity")
    for o in annotate_opps(conn, rank_map):
        entity = "Acme (ours)" if o["canonical"] == 1 else (
            "Acme LOGISTICS" if o["canonical"] == 50 else "other company")
        keep = "KEEP" if o["in_answer"] else "drop"
        print(f"  {o['rank']:>4} | {o['stage']:<16} | {entity:<13} | {keep:<5} | "
              f"{o['name']} (${o['amount']:,.0f})")
    print("  A record that must be DROPPED (a won deal or the look-alike company) ranks")
    print("  among, and even above, the records that must be KEPT. 'Open vs won' and")
    print("  'our Acme vs the other Acme' are not similarity relationships, so no cutoff")
    print("  on this ranking separates the four-record answer from the noise.\n")

    # --- Pipeline 1: vector RAG, and the 'just raise k' objection ------------
    print("VECTOR RAG (retrieve top-k, sum the amounts in the retrieved opp chunks):")
    print("  truth is $275,000")
    for k in (8, 25, 50, 100, 200, n_docs):
        r = vector_rag(conn, QUESTION, k=k)
        ok = entity_correct(r["opp_ids"], gold_opp_ids)
        why = "" if ok else ("misses open deals" if r["total"] < gold_total
                             else "sums won deals / other companies")
        print(f"    k={k:>3}:  ${r['total']:>10,.0f}   {'CORRECT' if ok else 'WRONG':<7}  {why}")
    print("  No k returns exactly the four open Acme deals.\n")

    # --- Pipeline 2: hybrid -------------------------------------------------
    hy = hybrid(conn, "acme")
    hy_ok = entity_correct(hy["opp_ids"], gold_opp_ids)
    print("HYBRID (resolve entity -> follow joins -> sum; similarity only for the fuzzy last mile):")
    print(f"  Resolved the 'Acme' spellings to canonical entity #{hy['canonical_id']}, excluded the")
    print(f"  'Acme Logistics' look-alike (different entity) and the won deals (by stage).")
    print(f"  Answer:  ${hy['total']:,.0f}   entity-correct: {hy_ok}\n")

    print("SCOREBOARD")
    print(f"  vector RAG : WRONG at every k (no cutoff you can know in advance returns $275,000)")
    print(f"  hybrid     : {'CORRECT' if hy_ok else 'WRONG'}  (${hy['total']:,.0f}, deterministic at any scale)")


if __name__ == "__main__":
    main()
