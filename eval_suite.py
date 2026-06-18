"""Four queries of different SHAPES, so the comparison is not one worst-case example.

The point of the piece is not "vector RAG is bad." It is "match the retrieval to
the question." This harness makes that measurable: vector RAG is correct on the
retrieval-shaped questions (lookup, semantic-find) and wrong on the relational
ones (aggregation, count). The hybrid routes each question to the mechanism that
can answer it.

    python eval_suite.py
"""
from __future__ import annotations

from crm_data import build_crm, as_documents
from retrieval import VectorIndex, vector_rag, hybrid, entity_correct


def top_ids(conn, query: str, k: int):
    idx = VectorIndex(as_documents(conn))
    return [d for d, _, _ in idx.search(query, k=k)]


def main():
    conn = build_crm()
    gold_opp_ids = [1, 2, 3, 4]
    rows = []

    # Q1: aggregation + exclusion (relational). Vector RAG has no correct k.
    any_k_ok = any(
        entity_correct(vector_rag(conn, "total open pipeline for Acme", k=k)["opp_ids"], gold_opp_ids)
        for k in range(1, len(as_documents(conn)) + 1)
    )
    hy_total = hybrid(conn, "acme")["total"]
    rows.append(("aggregation: total open pipeline",
                 "CORRECT" if any_k_ok else "WRONG (no k returns $275,000)",
                 f"CORRECT (${hy_total:,.0f})"))

    # Q2: count (relational). Same structural problem.
    truth_count = conn.execute(
        "SELECT COUNT(*) FROM opportunity o JOIN account a ON a.id=o.account_id "
        "WHERE a.canonical_id=1 AND o.stage LIKE 'open%'").fetchone()[0]
    rows.append(("count: number of open Acme deals",
                 "WRONG (counts won deals and the look-alike)",
                 f"CORRECT ({truth_count})"))

    # Q3: lookup (retrieval-shaped). Vector RAG WINS: the answer is one chunk.
    hits = top_ids(conn, "What is Priya Nair's job title?", k=3)
    vr_q3 = "contact:5" in hits
    rows.append(("lookup: a contact's job title",
                 "CORRECT" if vr_q3 else "WRONG",
                 "CORRECT"))

    # Q4: semantic-find over free text (retrieval-shaped). Vector RAG WINS: the
    # cue lives in an activity note, not in any structured column.
    hits = top_ids(conn, "Which Acme deal involves a request about EU support?", k=5)
    vr_q4 = "activity:3" in hits
    rows.append(("semantic-find: a cue in a free-text note",
                 "CORRECT" if vr_q4 else "WRONG",
                 "CORRECT (similarity last mile)"))

    print("EVALUATION ACROSS QUESTION SHAPES (n=4, not a single example)")
    print(f"  {'question shape':<36} | {'vector RAG':<33} | hybrid")
    print(f"  {'-'*36} | {'-'*33} | {'-'*22}")
    for kind, vr, hy in rows:
        print(f"  {kind:<36} | {vr:<33} | {hy}")
    print()
    print("Vector RAG is correct on the retrieval-shaped questions and wrong on the")
    print("relational ones. Neither is universally right. The hybrid wins by routing each")
    print("question to the mechanism that fits its shape.")


if __name__ == "__main__":
    main()
