"""
03_NotebookLM_RAG_Audit.py
Week 5 — Strict Source-Grounding & Global Sensemaking Audit
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import re, json

# ---------- Data models ----------

@dataclass
class Source:
    doc_id: str
    chunks: List[str]

@dataclass
class Citation:
    span: str
    doc_id: str
    chunk_idx: int

@dataclass
class Response:
    text: str
    citations: List[Citation] = field(default_factory=list)

@dataclass
class Query:
    text: str

@dataclass
class Probe:
    query: str               # should NOT be answerable from sources

@dataclass
class AudioOverview:
    episode_id: str
    transcript: str

# ---------- 1. Citation verifier ----------

def _tokenize(s: str) -> set:
    return set(re.findall(r"\w+", s.lower()))

def verify_span(citation: Citation, sources: Dict[str, Source]) -> float:
    """Return 0..1 support: Jaccard overlap between claim and cited chunk."""
    src = sources.get(citation.doc_id)
    if not src or citation.chunk_idx >= len(src.chunks):
        return 0.0
    claim_tok = _tokenize(citation.span)
    chunk_tok = _tokenize(src.chunks[citation.chunk_idx])
    if not claim_tok:
        return 0.0
    return len(claim_tok & chunk_tok) / len(claim_tok)

def verify_citations(resp: Response, sources, log, threshold=0.5):
    for cite in resp.citations:
        support = verify_span(cite, sources)
        status = "✓" if support >= threshold else "✗ DECORATIVE"
        log.append(f"[CITE] claim='{cite.span[:40]}' "
                   f"span={cite.doc_id}/chunk#{cite.chunk_idx} "
                   f"support={support:.2f} {status}")
        yield cite, support

# ---------- 2. Negative-constraint probes ----------

REFUSAL_PATTERNS = [
    r"\bnot (mentioned|specified|provided|in the (documents|sources))\b",
    r"\bcannot find\b", r"\bno information\b",
    r"\bthe sources do not\b", r"\bnot covered\b",
]

def is_refusal(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in REFUSAL_PATTERNS)

def run_refusal_probes(endpoint, probes: List[Probe], log) -> Dict:
    refused = 0
    for i, p in enumerate(probes, 1):
        resp = endpoint.ask(p.query)
        ok = is_refusal(resp.text)
        refused += int(ok)
        mark = "✓" if ok else "✗ PARAMETRIC LEAK"
        log.append(f"[PROBE-{i:02d}] q='{p.query[:40]}' → "
                   f"{'REFUSED' if ok else 'ANSWERED'} {mark}")
    return {"n": len(probes), "refused": refused,
            "pass_rate": refused / max(len(probes), 1)}

# ---------- 3. Audio Overview fidelity ----------

def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

def supported_by(sentence: str, sources, threshold=0.6) -> bool:
    sent_tok = _tokenize(sentence)
    if not sent_tok:
        return True
    best = 0.0
    for src in sources.values():
        for chunk in src.chunks:
            c = _tokenize(chunk)
            if c:
                best = max(best, len(sent_tok & c) / len(sent_tok))
    return best >= threshold

def audit_audio(ep: AudioOverview, sources, log) -> Dict:
    flagged = 0
    for sent in split_sentences(ep.transcript):
        if not supported_by(sent, sources):
            flagged += 1
            log.append(f"[{ep.episode_id}] sent='{sent[:50]}...' "
                       f"→ UNSUPPORTED ✗")
    total = len(split_sentences(ep.transcript))
    return {"episode": ep.episode_id, "sentences": total,
            "unsupported": flagged,
            "fidelity": 1 - flagged / max(total, 1)}

# ---------- Orchestrator ----------

def run_audit(endpoint, queries, probes, audio_eps, sources):
    log = []
    cite_stats = {"checked": 0, "supported": 0}
    for q in queries:
        resp = endpoint.ask(q.text)
        for cite, score in verify_citations(resp, sources, log):
            cite_stats["checked"] += 1
            cite_stats["supported"] += int(score >= 0.5)

    refusal_stats = run_refusal_probes(endpoint, probes, log)
    audio_stats = [audit_audio(ep, sources, log) for ep in audio_eps]

    return {
        "citation": {**cite_stats,
                     "pass_rate": cite_stats["supported"] /
                                   max(cite_stats["checked"], 1)},
        "refusal": refusal_stats,
        "audio": audio_stats,
        "log": log,
    }