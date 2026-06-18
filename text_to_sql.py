"""The other popular answer: let the LLM write SQL. It fails differently.

Text-to-SQL is not strawmanned here. The model writes valid, plausible SQL. The
problem is that the two facts it needs are not visible in the schema:

  1. the STAGE ENCODING. Stages are 'open:negotiation', 'open:proposal', etc.,
     not the literal 'open' a model reasonably guesses. The guess returns nothing.
  2. ENTITY RESOLUTION. The four Acme spellings (and the trap "Acme Logistics",
     a different company) are not distinguishable by name matching. A
     `name LIKE '%Acme%'` filter silently folds the wrong company in.

These are the two most common text-to-SQL outputs for this question. A live
model emits variants of the same two errors. Each query below is run for real
against the CRM so you see the actual wrong number, not a claimed one.
"""
from __future__ import annotations

from crm_data import build_crm
from demo_problem import ground_truth


GUESSES = [
    ("Guess 1: literal stage value the model cannot see is encoded differently",
     """
     SELECT COALESCE(SUM(o.amount), 0)
     FROM opportunity o JOIN account a ON a.id = o.account_id
     WHERE a.name LIKE '%Acme%' AND o.stage = 'open'
     """),
    ("Guess 2: fixes the stage, but name-matching folds in 'Acme Logistics'",
     """
     SELECT COALESCE(SUM(o.amount), 0)
     FROM opportunity o JOIN account a ON a.id = o.account_id
     WHERE a.name LIKE '%Acme%' AND o.stage LIKE 'open%'
     """),
]


def main():
    conn = build_crm()
    truth = ground_truth(conn)["total"]
    print(f"CORRECT ANSWER: ${truth:,.0f}\n")
    for label, sql in GUESSES:
        got = conn.execute(sql).fetchone()[0]
        delta = got - truth
        tag = "CORRECT" if abs(delta) < 1e-6 else "WRONG"
        print(f"{label}")
        print(f"  -> ${got:,.0f}   {tag}   ({'off by $%s' % f'{delta:,.0f}' if delta else 'exact'})")
        print()
    print("Both queries are valid SQL. Both are wrong, for reasons that live in")
    print("data semantics the model never sees: stage encoding and entity identity.")
    print("The correct query needs the canonical_id resolution and the 'open%' prefix,")
    print("neither of which is guessable from column names alone.")


if __name__ == "__main__":
    main()
