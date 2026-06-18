"""A small, synthetic, deliberately messy CRM at semi-realistic volume.

The mess is the point. Real business records are relational and dirty in ways
that document-style retrieval never sees:

  * the SAME company appears under four spellings and two legal entities,
  * a different company shares a token in its name ("Acme Logistics"), a trap
    for any name-substring match,
  * a single customer's pipeline spans multiple accounts and tables, and the
    answer to an ordinary question lives in the JOINS, not in any one record.

Everything here is fabricated with a fixed seed (no real customer data). The
schema mirrors the shape of Dynamics 365 / Dataverse sales records without
copying any of it. The hero records (the "Acme" group) are hand-authored for a
clear narrative; the rest is generated filler so the corpus is large enough that
no top-k similarity cutoff can isolate the right relational subset by luck.
"""
from __future__ import annotations

import random
import sqlite3

SCHEMA = """
CREATE TABLE account (
    id            INTEGER PRIMARY KEY,
    name          TEXT NOT NULL,     -- as typed by whoever created the record
    parent_id     INTEGER,           -- self-reference: subsidiary -> parent
    canonical_id  INTEGER,           -- the resolved "real" entity this row belongs to
    country       TEXT
);
CREATE TABLE contact (
    id          INTEGER PRIMARY KEY,
    account_id  INTEGER NOT NULL,
    name        TEXT NOT NULL,
    title       TEXT,
    email       TEXT
);
CREATE TABLE opportunity (
    id            INTEGER PRIMARY KEY,
    account_id    INTEGER NOT NULL,
    primary_contact_id INTEGER,
    name          TEXT NOT NULL,
    stage         TEXT NOT NULL,     -- 'open:*' stages vs 'won'/'lost'
    amount        REAL NOT NULL
);
CREATE TABLE line_item (
    id              INTEGER PRIMARY KEY,
    opportunity_id  INTEGER NOT NULL,
    product         TEXT NOT NULL,
    amount          REAL NOT NULL
);
CREATE TABLE activity (
    id              INTEGER PRIMARY KEY,
    opportunity_id  INTEGER,
    account_id      INTEGER,
    kind            TEXT,
    note            TEXT
);
"""

# ---- hero records: the Acme group + the named distractors -------------------
# canonical_id 1 ties the four Acme spellings/entities together; a naive system
# has no idea they are the same. Acme Logistics (canonical 50) is a DIFFERENT
# company that merely shares the "Acme" token.
HERO_ACCOUNTS = [
    # id, name,                 parent_id, canonical_id, country
    (1, "Acme Corp",            None, 1,  "US"),
    (2, "ACME Corporation",     None, 1,  "US"),
    (3, "Acme Inc.",            None, 1,  "US"),
    (4, "Acme Robotics GmbH",   1,    1,  "DE"),
    (50, "Acme Logistics",      None, 50, "US"),   # different company, shared token
    (51, "Globex",              None, 51, "US"),
    (52, "Initech LLC",         None, 52, "US"),
]

HERO_CONTACTS = [
    (1, 1, "Dana Reyes",  "VP Operations",    "dana.reyes@acme.example"),
    (2, 2, "Dana Reyes",  "VP Operations",    "dana@acme-corp.example"),
    (3, 4, "Lukas Bauer", "Procurement Lead", "l.bauer@acme-robotics.example"),
    (4, 50, "Sam Okafor", "Logistics Manager","sam@acme-logistics.example"),
    (5, 3, "Priya Nair",  "Finance Director", "priya.nair@acme.example"),
]

# The four OPEN deals (120+45+30+80 = 275k) are the answer. The two WON Acme
# deals share the "Acme" token, so similarity ranks them right next to the open
# ones: any k big enough to catch all four open deals also catches the won ones.
HERO_OPPS = [
    # id, account_id, primary_contact_id, name,                      stage,             amount
    (1, 1, 1, "Acme platform renewal",          "open:negotiation", 120000.0),
    (2, 2, 2, "Acme add-on licenses",           "open:proposal",     45000.0),
    (3, 3, 5, "Acme finance module",            "open:qualify",      30000.0),
    (4, 4, 3, "Acme Robotics line expansion",   "open:proposal",     80000.0),
    (5, 1, 1, "Acme 2024 platform project",     "won",              200000.0),  # excluded: won
    (6, 2, 2, "Acme legacy data migration",     "won",              150000.0),  # excluded: won
    (7, 50, 4, "Acme Logistics fleet automation","open:proposal",    60000.0),  # excluded: different company
    (8, 51, None, "Globex cloud migration",     "open:negotiation",  95000.0),  # excluded: different company
]

HERO_LINES = [
    (1, 1, "Platform subscription", 90000.0),
    (2, 1, "Premier support",       30000.0),
    (3, 2, "Seat licenses",         45000.0),
    (4, 3, "Finance module",        30000.0),
    (5, 4, "Robotics connector",    50000.0),
    (6, 4, "Onboarding services",   30000.0),
]

HERO_ACTIVITIES = [
    (1, 1, 1, "meeting", "QBR with Dana; renewal verbal yes, paperwork in legal"),
    (2, 2, 2, "email",   "Sent Acme add-on quote to Dana"),
    (3, 4, 4, "call",    "Lukas wants the Acme expansion bundled with EU support SLA"),
    (4, 3, 3, "meeting", "Priya reviewing Acme finance module ROI"),
]

# ---- generated filler so the corpus is realistically large ------------------
_COMPANIES = [
    "Northwind Traders", "Contoso", "Fabrikam", "Tailspin Toys", "Wingtip Toys",
    "Adventure Works", "Proseware", "Litware", "Wide World Importers", "Fourth Coffee",
    "Coho Vineyard", "Alpine Ski House", "Blue Yonder Airlines", "City Power and Light",
    "Consolidated Messenger", "Graphic Design Institute", "Humongous Insurance",
    "Lucerne Publishing", "Margie's Travel", "Nod Publishers", "Trey Research",
    "School of Fine Art", "Southridge Video", "Tasman Toys", "VanArsdel Ltd",
    "WideField Imaging", "Relecloud", "First Up Consultants", "Bellows College",
    "Best For You Organics", "Lamna Healthcare", "Munson Pickles", "Woodgrove Bank",
    "World Wide Importers", "Data Cloud Partners",
]
_PRODUCTS = ["Platform subscription", "Seat licenses", "Premier support",
             "Analytics module", "Integration connector", "Onboarding services",
             "Migration services", "Custom development", "Training package"]
_STAGES_OPEN = ["open:qualify", "open:proposal", "open:negotiation"]
_FIRST = ["Alex", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Jamie", "Avery", "Quinn", "Drew"]
_LAST = ["Smith", "Patel", "Nguyen", "Garcia", "Khan", "Muller", "Rossi", "Kim", "Silva", "Adams"]


def _generate(seed: int = 42):
    rng = random.Random(seed)
    accounts, contacts, opps, lines, acts = [], [], [], [], []
    aid, cid, oid, lid, actid = 100, 100, 100, 100, 100
    for company in _COMPANIES:
        canonical = aid
        accounts.append((aid, company, None, canonical, rng.choice(["US", "UK", "DE", "CA"])))
        contact_id = cid
        contacts.append((cid, aid, f"{rng.choice(_FIRST)} {rng.choice(_LAST)}",
                         rng.choice(["VP Sales", "CIO", "Director", "Buyer"]),
                         f"contact@{company.split()[0].lower()}.example"))
        cid += 1
        for _ in range(rng.randint(1, 4)):
            stage = rng.choice(_STAGES_OPEN + ["won", "lost"])
            amount = float(rng.choice([15000, 25000, 40000, 60000, 90000, 110000, 140000]))
            opps.append((oid, aid, contact_id, f"{company} {rng.choice(_PRODUCTS).lower()}", stage, amount))
            lines.append((lid, oid, rng.choice(_PRODUCTS), amount))
            lid += 1
            note = rng.choice([
                f"Discussed pricing on the {company} deal",
                f"Follow-up scheduled with {company}",
                f"{company} asked for a revised quote",
                f"Internal review of the {company} opportunity",
                f"Demo delivered to the {company} team",
            ])
            acts.append((actid, oid, aid, rng.choice(["call", "email", "meeting"]), note))
            actid += 1
            oid += 1
        aid += 1
    return accounts, contacts, opps, lines, acts


def build_crm(path: str = ":memory:", seed: int = 42) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    g_acc, g_con, g_opp, g_lin, g_act = _generate(seed)
    conn.executemany("INSERT INTO account VALUES (?,?,?,?,?)", HERO_ACCOUNTS + g_acc)
    conn.executemany("INSERT INTO contact VALUES (?,?,?,?,?)", HERO_CONTACTS + g_con)
    conn.executemany("INSERT INTO opportunity VALUES (?,?,?,?,?,?)", HERO_OPPS + g_opp)
    conn.executemany("INSERT INTO line_item VALUES (?,?,?,?)", HERO_LINES + g_lin)
    conn.executemany("INSERT INTO activity VALUES (?,?,?,?,?)", HERO_ACTIVITIES + g_act)
    conn.commit()
    return conn


def as_documents(conn: sqlite3.Connection) -> list[tuple[str, str]]:
    """Return (doc_id, text) chunks, one per record, the way naive RAG ingests a CRM."""
    docs: list[tuple[str, str]] = []
    for r in conn.execute("SELECT id, name, country FROM account"):
        docs.append((f"account:{r[0]}", f"Account: {r[1]} ({r[2]})"))
    for r in conn.execute("SELECT id, name, title, email FROM contact"):
        docs.append((f"contact:{r[0]}", f"Contact: {r[1]}, {r[2]} <{r[3]}>"))
    for r in conn.execute("SELECT id, name, stage, amount FROM opportunity"):
        docs.append((f"opp:{r[0]}", f"Opportunity: {r[1]}, stage {r[2]}, amount ${r[3]:,.0f}"))
    for r in conn.execute("SELECT id, product, amount FROM line_item"):
        docs.append((f"line:{r[0]}", f"Line item: {r[1]}, ${r[2]:,.0f}"))
    for r in conn.execute("SELECT id, kind, note FROM activity"):
        docs.append((f"activity:{r[0]}", f"Activity ({r[1]}): {r[2]}"))
    return docs


if __name__ == "__main__":
    conn = build_crm()
    counts = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
              for t in ("account", "contact", "opportunity", "line_item", "activity")}
    docs = as_documents(conn)
    print("CRM built:", ", ".join(f"{v} {k}s" for k, v in counts.items()))
    print(f"Flattened to {len(docs)} naive RAG document chunks.")
