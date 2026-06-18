"""The question, the correct answer, and why it lives in the joins.

Run:
    python demo_problem.py

An ordinary sales question: "What is our total OPEN pipeline for Acme across
every entity, and who are the key contacts?" The correct answer is not in any
single record. It requires (1) resolving four spellings and a subsidiary to one
customer, (2) following account -> opportunity -> line_item -> contact links,
and (3) excluding the won deal and the look-alike company. This script computes
the ground truth with the JOINs a correct system must perform, so later we can
measure what vector retrieval and text-to-SQL actually return against it.
"""
from __future__ import annotations

from crm_data import build_crm

QUESTION = ("What is our total OPEN pipeline for Acme across every entity, "
            "and who are the key contacts?")


def ground_truth(conn) -> dict:
    # canonical_id = 1 is the real "Acme" group: Acme Corp, ACME Corporation,
    # Acme Inc., and Acme Robotics GmbH. A correct system resolves all four.
    open_opps = conn.execute(
        """
        SELECT o.id, o.name, o.stage, o.amount, a.name
        FROM opportunity o
        JOIN account a ON a.id = o.account_id
        WHERE a.canonical_id = 1
          AND o.stage LIKE 'open%'
        ORDER BY o.amount DESC
        """
    ).fetchall()

    total = conn.execute(
        """
        SELECT COALESCE(SUM(o.amount), 0)
        FROM opportunity o
        JOIN account a ON a.id = o.account_id
        WHERE a.canonical_id = 1 AND o.stage LIKE 'open%'
        """
    ).fetchone()[0]

    contacts = conn.execute(
        """
        SELECT DISTINCT c.name, c.title
        FROM contact c
        JOIN account a ON a.id = c.account_id
        WHERE a.canonical_id = 1
        ORDER BY c.name
        """
    ).fetchall()

    return {"open_opps": open_opps, "total": total, "contacts": contacts}


if __name__ == "__main__":
    conn = build_crm()

    print(f"QUESTION:\n  {QUESTION}\n")

    print("THE TRAPS (why this is not a document lookup):")
    variants = conn.execute(
        "SELECT name, country FROM account WHERE canonical_id = 1 ORDER BY id"
    ).fetchall()
    print(f"  - 'Acme' is stored as {len(variants)} rows, two legal entities, "
          f"one real customer:")
    for name, country in variants:
        print(f"      {name}  ({country})")
    print("  - One $200,000 'Acme 2024 project' is WON, not open: it must be excluded.")
    print("  - 'Globex' has a $95,000 open deal and looks superficially similar: a distractor.\n")

    gt = ground_truth(conn)
    print("GROUND TRUTH (what a correct system must return):")
    print(f"  Total open pipeline for Acme: ${gt['total']:,.0f}")
    print("  Made of these open opportunities, scattered across 4 account rows:")
    for oid, name, stage, amount, acct in gt["open_opps"]:
        print(f"      ${amount:>9,.0f}  {name}  [{stage}]  on account '{acct}'")
    print("  Key contacts across all Acme entities:")
    for name, title in gt["contacts"]:
        print(f"      {name} ({title})")
    print()
    print("Note: the answer ($%s) is a SUM OVER JOINS across 4 resolved accounts." % f"{gt['total']:,.0f}")
    print("No single record contains it, so retrieving 'the most similar chunks' cannot produce it.")
