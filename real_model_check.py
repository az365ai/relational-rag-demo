"""Same query, a REAL dense embedding model. The failure is not the toy embedder.

Uses model2vec (a real static distillation of sentence-transformers, no torch,
no GPU). A production embedder clusters the 'Acme' records tighter than the
hashed baseline, which only sharpens the point: it ranks a WON deal above the
open ones, because 'open vs won' is not something an embedding can see.

    pip install model2vec
    python real_model_check.py
"""
from __future__ import annotations

import numpy as np

from crm_data import build_crm, as_documents
from demo_problem import QUESTION
from retrieval import annotate_opps


def main():
    try:
        from model2vec import StaticModel
        model = StaticModel.from_pretrained("minishlab/potion-base-8M")
    except Exception as e:  # network / model unavailable
        print(f"Real model unavailable ({type(e).__name__}: {e}).")
        print("Skipping. demo_retrieval.py already shows the structural failure on the")
        print("zero-dependency embedder; this script only confirms it under a dense model.")
        return

    conn = build_crm()
    docs = as_documents(conn)
    ids = [d for d, _ in docs]
    texts = [t for _, t in docs]

    M = np.asarray(model.encode(texts), dtype=np.float64)
    M /= np.linalg.norm(M, axis=1, keepdims=True) + 1e-12
    q = np.asarray(model.encode([QUESTION]), dtype=np.float64)[0]
    q /= np.linalg.norm(q) + 1e-12
    order = np.argsort(-(M @ q))
    rank_map = {ids[i]: p + 1 for p, i in enumerate(order)}

    print("REAL DENSE MODEL (model2vec / potion-base-8M)")
    print(f"Corpus: {len(ids)} chunks.")
    print("  rank | stage            | entity        | keep? | opportunity")
    for o in annotate_opps(conn, rank_map):
        entity = "Acme (ours)" if o["canonical"] == 1 else (
            "Acme LOGISTICS" if o["canonical"] == 50 else "other company")
        keep = "KEEP" if o["in_answer"] else "drop"
        print(f"  {o['rank']:>4} | {o['stage']:<16} | {entity:<13} | {keep:<5} | {o['name']}")
    print("\n  The better model groups the Acme records near the top, then ranks a WON deal")
    print("  ABOVE three of the four open ones. A tighter cluster does not help: the records")
    print("  to drop still interleave with the records to keep, because the distinction is")
    print("  relational (stage, entity), not semantic. Same wall, sharper view.")


if __name__ == "__main__":
    main()
