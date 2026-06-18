"""Three ways to answer the question, and what each actually returns.

1. vector_rag      - embed every record as an independent chunk, retrieve the
                     top-k most similar, naively aggregate. This is the standard
                     "put your data in a vector DB" pipeline.
2. hybrid          - resolve entities deterministically, follow the joins with a
                     real query, use similarity ONLY for the fuzzy last mile.

The embedding here is a zero-dependency hashed n-gram vector so the demo runs
with nothing but numpy. The failure it shows is STRUCTURAL, not a property of
this particular embedder: a production dense model retrieves a different set of
chunks but hits the same wall, because entity unification, the won/open
exclusion, and summation across joins are not similarity problems.
"""
from __future__ import annotations

import re
import hashlib
import numpy as np

from crm_data import build_crm, as_documents

DIM = 512


def _h(s: str) -> int:
    """Stable, process-independent hash (Python's built-in hash() is salted per
    run, which would make the demo non-reproducible)."""
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=4).digest(), "little")


def _embed(text: str) -> np.ndarray:
    """Hashed word + character-trigram bag, tf-weighted, L2-normalized."""
    v = np.zeros(DIM, dtype=np.float64)
    toks = re.findall(r"[a-z0-9]+", text.lower())
    for w in toks:
        v[_h("w:" + w) % DIM] += 1.0
        padded = f"#{w}#"
        for i in range(len(padded) - 2):
            v[_h("c:" + padded[i:i + 3]) % DIM] += 1.0
    n = np.linalg.norm(v)
    return v / n if n else v


def _cosine(q: np.ndarray, M: np.ndarray) -> np.ndarray:
    return M @ q


class VectorIndex:
    def __init__(self, docs):
        self.ids = [d for d, _ in docs]
        self.texts = [t for _, t in docs]
        self.M = np.vstack([_embed(t) for t in self.texts])

    def search(self, query: str, k: int = 6):
        scores = _cosine(_embed(query), self.M)
        order = np.argsort(-scores)[:k]
        return [(self.ids[i], self.texts[i], float(scores[i])) for i in order]


def vector_rag(conn, query: str, k: int = 6):
    """Standard pipeline: retrieve top-k chunks, then sum the dollar amounts that
    appear in the retrieved opportunity chunks. A faithful model of what a naive
    'ask the vector store and let the LLM add it up' agent does."""
    idx = VectorIndex(as_documents(conn))
    hits = idx.search(query, k=k)
    total = 0.0
    opp_ids = []
    for doc_id, text, _ in hits:
        if doc_id.startswith("opp:"):
            m = re.search(r"\$([\d,]+)", text)
            if m:
                total += float(m.group(1).replace(",", ""))
                opp_ids.append(int(doc_id.split(":")[1]))
    return {"hits": hits, "total": total, "opp_ids": opp_ids}


# ---- the hybrid -----------------------------------------------------------

def _canonical(conn, surface_name: str) -> int | None:
    """Deterministic entity resolution: map a surface spelling to the real
    customer. In production this is a resolution table / MDM key (here, the
    account.canonical_id Dynamics-style relationship), NOT an LLM guess."""
    row = conn.execute(
        "SELECT canonical_id FROM account WHERE lower(name) LIKE ? LIMIT 1",
        (f"%{surface_name.lower()}%",),
    ).fetchone()
    return row[0] if row else None


def hybrid(conn, entity_surface: str = "acme"):
    """Resolve the entity, then follow the joins with a real query. Similarity is
    not needed for this question at all; when it is (fuzzy product names, free
    text), it runs only after the relational scope is fixed."""
    canonical = _canonical(conn, entity_surface)
    rows = conn.execute(
        """
        SELECT o.id, o.name, o.amount
        FROM opportunity o
        JOIN account a ON a.id = o.account_id
        WHERE a.canonical_id = ? AND o.stage LIKE 'open%'
        """,
        (canonical,),
    ).fetchall()
    total = sum(r[2] for r in rows)
    return {"canonical_id": canonical, "opp_ids": [r[0] for r in rows], "total": total}


# ---- evaluation -----------------------------------------------------------

def entity_correct(retrieved_opp_ids, gold_opp_ids) -> bool:
    """The metric recall@k hides: did we get EXACTLY the right entity-resolved
    record set, no missing members, no wrong-entity or wrong-stage intruders."""
    return set(retrieved_opp_ids) == set(gold_opp_ids)


def similarity_ranks(conn, query: str, target_doc_ids):
    """Where do the answer records rank among ALL chunks?"""
    idx = VectorIndex(as_documents(conn))
    hits = idx.search(query, k=len(idx.ids))
    rank = {doc_id: i + 1 for i, (doc_id, _, _) in enumerate(hits)}
    return {t: rank.get(t) for t in target_doc_ids}, len(idx.ids)


def annotate_opps(conn, rank_map, hero_max_id: int = 8):
    """For the hand-authored 'Acme-ish' opportunities, line up the similarity rank
    against the two facts the embedder cannot see: stage (open vs won) and
    canonical entity (our Acme vs the look-alike). The interleaving this exposes
    is the structural failure, and it holds for any embedder."""
    canon = {r[0]: r[1] for r in conn.execute("SELECT id, canonical_id FROM account")}
    rows = conn.execute(
        "SELECT id, name, stage, amount, account_id FROM opportunity WHERE id <= ?",
        (hero_max_id,),
    ).fetchall()
    out = []
    for oid, name, stage, amount, aid in rows:
        c = canon[aid]
        out.append({
            "rank": rank_map.get(f"opp:{oid}"),
            "stage": stage,
            "canonical": c,
            "in_answer": (c == 1 and stage.startswith("open")),
            "name": name,
            "amount": amount,
        })
    return sorted(out, key=lambda d: (d["rank"] is None, d["rank"]))


def full_rank_map(conn, query: str):
    """rank of every chunk id, 1-based, by similarity to the query (hashed embedder)."""
    idx = VectorIndex(as_documents(conn))
    hits = idx.search(query, k=len(idx.ids))
    return {doc_id: i + 1 for i, (doc_id, _, _) in enumerate(hits)}
