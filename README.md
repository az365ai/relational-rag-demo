# relational-rag-demo

Your CRM is not a document store. This is a small, runnable demonstration that
**vector RAG and text-to-SQL both fail on relational business records**, and a
hybrid that does not.

The question is the kind a sales team asks every day:

> What is our total OPEN pipeline for Acme across every entity, and who are the
> key contacts?

The correct answer is **$275,000**. It is a sum over joins across four account
rows for one customer that is spelled four ways across two legal entities, with
a won deal and a look-alike company ("Acme Logistics") that must be excluded.
No single record contains the answer.

## Why retrieval misses it

Flatten the CRM to one text chunk per record, embed, and ask. Here is how
similarity ranks the "Acme" opportunities against the two facts the embedder
cannot see, stage (open vs won) and entity (our Acme vs the look-alike):

```
  rank | stage            | entity        | keep? | opportunity
     6 | open:negotiation | Acme (ours)   | KEEP  | Acme platform renewal
     7 | won              | Acme (ours)   | drop  | Acme legacy data migration
     8 | open:proposal    | Acme (ours)   | KEEP  | Acme add-on licenses
     9 | open:qualify     | Acme (ours)   | KEEP  | Acme finance module
    13 | open:proposal    | Acme (ours)   | KEEP  | Acme Robotics line expansion
    17 | open:proposal    | Acme LOGISTICS | drop  | Acme Logistics fleet automation
    19 | won              | Acme (ours)   | drop  | Acme 2024 platform project
```
*(a real dense embedding model, model2vec / potion-base-8M)*

A **won** deal ranks above three of the four open ones, and a different company
("Acme Logistics") ranks among them. "Open vs won" and "our Acme vs the other
Acme" are not similarity relationships, so no top-k cutoff separates the four
records you want from the records you must drop. Raising k does not help: small k
drops real deals, large k sums the won deals and other companies. There is no k
that returns $275,000.

Text-to-SQL fails differently: it writes valid SQL against semantics it cannot
see. A literal `stage = 'open'` matches nothing (stages are `open:proposal`,
etc.); a `name LIKE '%Acme%'` filter folds in the look-alike company. Both
queries run, both are wrong.

## The hybrid that works

Resolve the entity deterministically (a resolution key / `canonical_id`, not an
LLM guess), follow the joins with a real query, and use similarity only for the
fuzzy last mile. The LLM never guesses a JOIN. Result: **$275,000, exactly,
deterministic at any scale.**

The metric this surfaces is **entity-correct retrieval rate**, the share of
queries that retrieve exactly the right entity-resolved record set. `recall@k`
hides the failure; this does not.

## Run it

```bash
pip install numpy            # the only hard dependency
python demo_problem.py       # the question, the traps, and the ground truth
python demo_retrieval.py     # vector RAG vs the hybrid, with the k-sweep
python text_to_sql.py        # the two plausible-but-wrong SQL guesses
python eval_suite.py         # all four question shapes: where each method wins

pip install model2vec        # optional: confirm the failure under a real dense model
python real_model_check.py

python -m unittest discover -s tests -v
```

Everything is fabricated with a fixed seed. No real customer data. The schema
mirrors the shape of Dynamics 365 / Dataverse sales records.

## Files

| file | what it is |
|---|---|
| `crm_data.py` | the synthetic, deliberately messy CRM (four spellings, a subsidiary, a look-alike, generated filler) |
| `demo_problem.py` | the question and the ground-truth answer via correct joins |
| `retrieval.py` | the embedder, the vector-RAG pipeline, the hybrid, and the metric |
| `demo_retrieval.py` | the head-to-head, including the "just raise k" sweep |
| `eval_suite.py` | all four question shapes, where each method wins and loses |
| `text_to_sql.py` | two plausible text-to-SQL guesses, run for real |
| `real_model_check.py` | the same failure under a production dense model |
| `tests/` | proofs of every claim above |

## License

MIT. Written up at [az365.ai](https://az365.ai).
