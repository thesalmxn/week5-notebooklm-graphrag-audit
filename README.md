# Week 5 — Strict Source-Grounding & Global Sensemaking Audit

**Topic:** The Google NotebookLM Audit & GraphRAG
**Course:** Conversational Artificial Intelligence — ps05: Dual-Path GraphRAG & The Google NotebookLM Grounding Lab
**Author:** Mohammad Salman Sunakal (5019034)

---

## What this is

An **audit harness** for grounded RAG systems. It tests whether a retrieval-augmented
endpoint actually grounds its answers in the source documents — or whether it
hallucinates from parametric memory while attaching citations that look real.

Three independent audit passes:

| # | Pass | Question it answers |
|---|------|---------------------|
| 1 | **Citation verification** | Do cited spans actually support the claim? |
| 2 | **Negative-constraint probes** | Does the endpoint refuse out-of-corpus queries? |
| 3 | **Audio Overview fidelity** | Does the 2-host podcast invent facts? |

The harness is endpoint-agnostic: it can be pointed at any RAG system that
implements the `ask()` / `audio_overviews()` contract. A `MockEndpoint` is
included so the audit is reproducible without a live API.

---

## Files
03_NotebookLM_RAG_Audit.py # audit harness (the implementation)
ps05_solution.ipynb # runs the harness, produces the numbers
Week5_NotebookLM_GraphRAG_Audit.pptx # 5-slide lightning presentation

---

## Why a mock endpoint

Google NotebookLM is closed-source and exposes no public API — it cannot be
called from Python. The audit therefore targets the **output contract** that a
grounded system like NotebookLM promises, not its internal weights.

`MockEndpoint` simulates three behaviors so the harness can demonstrate its own
pass/fail logic:

- `mixed` (default) — 30% parametric leak, 20% decorative citation, 50% grounded
- `refuse_ok` — always refuses (strict closed-world behavior)
- `grounded` — always grounded (baseline)

The harness is the deliverable. The mock is a test fixture. Pointing the harness
at a real endpoint requires only implementing `ask()` and `audio_overviews()`.

---

## How to run

### Requirements

- Python 3.10+
- Jupyter / VS Code notebook support

### Steps

1. Place `03_NotebookLM_RAG_Audit.py` and `ps05_solution.ipynb` in the **same folder**.
2. Open `ps05_solution.ipynb`.
3. Run all cells top-to-bottom (**Restart & Run All**).

The notebook loads the audit module via `importlib` (the filename begins with
`03_`, so it is not importable as a normal module), builds a mock corpus, runs
the three passes, and prints the results table and audit log.

---

## Results (from `ps05_solution.ipynb`)

| Probe Type | N | Pass Rate | Failure Caught |
|---|---|---|---|
| Citation support | 28 | **86%** | Decorative citations |
| Negative-constraint | 25 | **0% → 100%** | Parametric leakage / fabrication |
| Audio fidelity | 3 | **67%** | Creative synthesis → false facts |

**Interpretation**

- **Citation support — 86%:** 14% of citations were *decorative* — the claim had
  zero token overlap with the cited chunk. The verifier caught them.
- **Negative-constraint — 0% → 100%:** the baseline endpoint refused **0%** of
  out-of-corpus probes (it hallucinated every time); the hardened endpoint
  refused **100%**. Same harness, opposite verdicts — proof the detector works.
- **Audio fidelity — 67%:** 2 of 3 podcast sentences were grounded. The
  unsupported sentence ("Founded in 1998") contained a fact absent from the
  source documents.

---

## Design notes

- **`verify_span()`** returns a 0–1 support score using token-overlap (Jaccard
  on claim tokens). Threshold for "supported" is `0.5`.
- **`is_refusal()`** matches a small set of refusal phrases
  (`"not mentioned in the sources"`, `"cannot find"`, `"no information"`, …).
  This is intentionally conservative — false negatives (missed refusals) are
  preferred to false positives.
- **`supported_by()`** checks each podcast sentence against the whole corpus
  with a `0.6` overlap threshold.
- The harness is deterministic: `MockEndpoint` seeds `random` with `42`, so
  results are reproducible.

---

## GraphRAG (context)

Vector RAG returns top-*k* locally similar chunks and cannot answer
corpus-wide questions such as *"what are the recurring themes across all
documents?"* GraphRAG addresses this by building a knowledge graph, applying
**Leiden community detection**, and generating hierarchical community summaries
so the system can answer from global structure rather than local chunks.

The audit's corpus-level probe contrasts:

- vector RAG top-*k* → 3 disjoint chunks, no synthesis
- Leiden community summary → 1 coherent theme answer

---

## Takeaway

> Grounded RAG is not retrieval + generation.
> It is retrieval + generation **constrained by refusal**
> + a verifiable citation lineage.
> Without a refusal path, citations are decoration.

---

## License

Course submission — not for redistribution.