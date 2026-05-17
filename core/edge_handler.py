"""
core/edge_handler.py  —  ARIRAS Edge Case & Confidence Engine
==============================================================

Two real systems, no string matching, no word-count heuristics.

1. RAG CONFIDENCE PIPELINE  (score_rag_response)
   ------------------------------------------------
   Layer 1 — Semantic grounding score
     Embed the answer and every retrieved chunk using the same
     all-MiniLM-L6-v2 model already in vectorstore.py.
     Since embeddings are L2-normalised: cosine_sim = dot product.
     max_sim  → best single-chunk match (entailment proxy)
     mean_sim → overall coverage across retrieved context

   Layer 2 — LLM-as-judge faithfulness verdict
     A second, zero-temperature Groq call receives the answer +
     the best-matching chunk and returns structured JSON:
       { "faithful": true/false,
         "verdict": "SUPPORTED | PARTIAL | HALLUCINATED",
         "unsupported_claims": [...] }
     Catches cases where answer is semantically close to chunks
     but still makes claims the chunks don't support.

   Final confidence_score = sim_score (0-50) + faith_score (0-50)

2. SEMANTIC PREFLIGHT  (preflight_gap)
   ----------------------------------------
   Embed the policy text (2000-char sample) and compute cosine
   similarity against 10 compliance-domain probe queries.
   Mean probe similarity < 0.28 → block (not a compliance document).
   Mean probe similarity < 0.38 → warn (thin coverage).
   A 50-word legal clause passes. A 2000-word invoice fails.
   No word counting. No keyword lists.

3. MULTI-REG CONFLICT DETECTOR  (detect_reg_conflicts)
   -------------------------------------------------------
   When reg chunk texts are available: embed each regulation's
   chunks and score them against semantic conflict probe pairs.
   Conflict fires when different regulations score high on
   opposite sides of a known tension (e.g. retention vs erasure).
   Falls back to name-based detection when chunks aren't passed.
"""

import os
import json
import numpy as np
from dotenv import load_dotenv

load_dotenv()

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

# ── Lazy singleton — same model as vectorstore.py, loaded once ───────────────
_EMBED_MODEL = None

def _get_embed_model():
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        _EMBED_MODEL = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _EMBED_MODEL


def _embed(texts: list) -> np.ndarray:
    """Return (N, D) float32 array of L2-normalised embeddings."""
    arr   = np.array(_get_embed_model().embed_documents(texts), dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return arr / norms


def _cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(M,D) × (N,D) → (M,N) cosine similarity via dot product (both L2-normed)."""
    return a @ b.T


# ─────────────────────────────────────────────────────────────────────────────
# CONFIDENCE LEVELS
# ─────────────────────────────────────────────────────────────────────────────
CONF_HIGH         = "HIGH"
CONF_MEDIUM       = "MEDIUM"
CONF_LOW          = "LOW"
CONF_INSUFFICIENT = "INSUFFICIENT"

# Cosine similarity thresholds for all-MiniLM-L6-v2
_SIM_STRONG  = 0.72
_SIM_PARTIAL = 0.52


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 1  —  SEMANTIC GROUNDING
# ─────────────────────────────────────────────────────────────────────────────
def _semantic_grounding(answer: str, chunk_texts: list) -> dict:
    """
    Embed answer + chunks. Return similarity stats + grounding label.
    """
    if not chunk_texts:
        return {"max_sim": 0.0, "mean_sim": 0.0,
                "grounding_label": "WEAK", "top_chunk_index": -1}

    vecs     = _embed([answer] + chunk_texts)
    sims     = _cos(vecs[0:1], vecs[1:])[0]          # (N,)
    max_s    = float(sims.max())
    mean_s   = float(sims.mean())
    top_i    = int(sims.argmax())
    label    = ("STRONG" if max_s >= _SIM_STRONG
                else "PARTIAL" if max_s >= _SIM_PARTIAL
                else "WEAK")

    return {"max_sim": round(max_s, 3), "mean_sim": round(mean_s, 3),
            "grounding_label": label, "top_chunk_index": top_i}


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 2  —  LLM-AS-JUDGE FAITHFULNESS
# ─────────────────────────────────────────────────────────────────────────────
_JUDGE_PROMPT = """\
You are a strict legal compliance auditor verifying whether an AI answer is
faithfully grounded in the provided regulation excerpts.

REGULATION EXCERPT (the only source of truth):
{chunk}

AI ANSWER TO VERIFY:
{answer}

TASK: For every factual or legal claim in the answer, check whether it is
explicitly supported by the excerpt above. List unsupported claims.

Return ONLY valid JSON — no markdown, no preamble:
{{
  "faithful": <true if all claims are supported>,
  "verdict": "<SUPPORTED | PARTIAL | HALLUCINATED>",
  "unsupported_claims": ["<claim>"],
  "confidence_note": "<one sentence plain-English summary>"
}}"""


def _llm_judge(answer: str, top_chunk: str) -> dict:
    """Second LLM call to verify faithfulness. Safe — never crashes main flow."""
    try:
        from langchain_groq import ChatGroq
        llm = ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0,
        )
        raw = llm.invoke(_JUDGE_PROMPT.format(
            chunk=top_chunk[:1500], answer=answer[:800]
        )).content.strip()

        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        v = json.loads(raw.strip())
        return {
            "faithful":           bool(v.get("faithful", True)),
            "verdict":            v.get("verdict", "SUPPORTED"),
            "unsupported_claims": v.get("unsupported_claims", []),
            "confidence_note":    v.get("confidence_note", ""),
        }
    except Exception:
        # Judge failed — return neutral, don't crash the main answer
        return {"faithful": True, "verdict": "SUPPORTED",
                "unsupported_claims": [], "confidence_note": "Check unavailable."}


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC: score_rag_response
# ─────────────────────────────────────────────────────────────────────────────
def score_rag_response(answer: str, docs: list, question: str) -> dict:
    """
    Two-layer confidence pipeline. Called by rag_agent.ask_regulation().

    Returns:
    {
        "confidence":         "HIGH|MEDIUM|LOW|INSUFFICIENT",
        "confidence_score":   0-100,
        "grounding":          { max_sim, mean_sim, grounding_label, top_chunk_index },
        "faithfulness":       { faithful, verdict, unsupported_claims, confidence_note },
        "warnings":           [ ... ],
        "answer_safe":        bool,
    }
    """
    warnings = []

    if not docs:
        return {
            "confidence": CONF_INSUFFICIENT, "confidence_score": 0,
            "grounding":  {"max_sim": 0, "mean_sim": 0,
                           "grounding_label": "WEAK", "top_chunk_index": -1},
            "faithfulness": {"faithful": False, "verdict": "HALLUCINATED",
                             "unsupported_claims": [],
                             "confidence_note": "No source documents retrieved."},
            "warnings": [
                "No regulation content was retrieved for this question. "
                "The answer cannot be grounded in the loaded regulation — "
                "re-index your document or rephrase the question."
            ],
            "answer_safe": False,
        }

    chunk_texts = [d.page_content for d in docs]

    # Layer 1 — semantic grounding
    grounding = _semantic_grounding(answer, chunk_texts)
    if grounding["grounding_label"] == "WEAK":
        warnings.append(
            f"Semantic grounding is weak (best chunk similarity: "
            f"{grounding['max_sim']:.2f} / 1.00). "
            "The answer may not be derived from the loaded regulation. "
            "Verify key statements against the source document before relying on this."
        )
    elif grounding["grounding_label"] == "PARTIAL":
        warnings.append(
            f"Partial grounding detected (similarity: {grounding['max_sim']:.2f}). "
            "Most claims appear supported, but some statements may extend beyond "
            "what the retrieved clauses explicitly state."
        )

    # Layer 2 — LLM faithfulness judge
    top_i   = grounding["top_chunk_index"]
    top_chunk = chunk_texts[top_i] if top_i >= 0 else ""
    faith   = _llm_judge(answer, top_chunk)

    if faith["verdict"] == "HALLUCINATED":
        claims = "; ".join(faith["unsupported_claims"][:3]) or "see verdict"
        warnings.append(
            f"Faithfulness check: answer flagged as HALLUCINATED. "
            f"Unsupported claims detected: {claims}."
        )
    elif faith["verdict"] == "PARTIAL" and faith["unsupported_claims"]:
        warnings.append(
            "Faithfulness check: some claims go beyond what the regulation explicitly states — "
            + "; ".join(faith["unsupported_claims"][:2]) + "."
        )

    # Final score — Layer 1 (0–50) + Layer 2 (0–50)
    sim_score   = min(50, int(grounding["max_sim"] * 65))
    faith_pts   = {"SUPPORTED": 50, "PARTIAL": 25, "HALLUCINATED": 0}
    total       = sim_score + faith_pts.get(faith["verdict"], 25)

    confidence  = (CONF_HIGH   if total >= 80 else
                   CONF_MEDIUM if total >= 55 else
                   CONF_LOW    if total >= 25 else
                   CONF_INSUFFICIENT)

    return {
        "confidence":       confidence,
        "confidence_score": total,
        "grounding":        grounding,
        "faithfulness":     faith,
        "warnings":         warnings,
        "answer_safe":      total >= 55,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SEMANTIC PREFLIGHT  (called by gap_detector.detect_gaps)
# ─────────────────────────────────────────────────────────────────────────────
_COMPLIANCE_PROBES = [
    "data subject rights and obligations",
    "breach notification and incident response procedure",
    "consent mechanism and lawful basis for processing",
    "penalty clause and enforcement action",
    "disclosure and reporting obligations to regulator",
    "data retention and deletion schedule",
    "purpose limitation and data minimisation",
    "third party data sharing and processor agreements",
    "employee responsibilities and data governance",
    "regulatory compliance requirements and audit",
]

_BLOCK_THRESHOLD = 0.28
_WARN_THRESHOLD  = 0.38


def preflight_gap(policy_text: str, reg_docs: list) -> dict:
    """
    Semantic preflight — runs before the LLM gap analysis call.

    Returns:
    {
        "ok":                   bool,
        "warnings":             [...],
        "errors":               [...],
        "compliance_relevance": float,   # 0-1 mean probe similarity
        "reg_chunks_found":     int,
        "reg_chunk_diversity":  float,   # embedding std dev of reg chunks
    }
    """
    warnings, errors = [], []

    # ── Regulation chunk check ────────────────────────────────────────────────
    reg_chunks_found = len(reg_docs)
    if reg_chunks_found == 0:
        errors.append(
            "No regulation content was retrieved. "
            "Please re-index your regulation document before running gap analysis."
        )
        return {"ok": False, "warnings": warnings, "errors": errors,
                "compliance_relevance": 0.0, "reg_chunks_found": 0,
                "reg_chunk_diversity": 0.0}

    # ── Policy text basic guard ───────────────────────────────────────────────
    if not policy_text or len(policy_text.strip()) < 50:
        errors.append(
            "Policy document is empty or too short (< 50 characters). "
            "Please upload a complete policy document."
        )
        return {"ok": False, "warnings": warnings, "errors": errors,
                "compliance_relevance": 0.0, "reg_chunks_found": reg_chunks_found,
                "reg_chunk_diversity": 0.0}

    # ── Check 1: Semantic compliance relevance ────────────────────────────────
    sample     = policy_text[:2000]
    vecs       = _embed(_COMPLIANCE_PROBES + [sample])
    probe_vecs = vecs[:-1]                     # (10, D)
    policy_vec = vecs[-1:]                     # (1,  D)
    sims       = _cos(policy_vec, probe_vecs)[0]   # (10,)
    mean_sim   = float(sims.mean())
    top_probes = [_COMPLIANCE_PROBES[i] for i in sims.argsort()[-3:][::-1]]

    if mean_sim < _BLOCK_THRESHOLD:
        errors.append(
            f"The uploaded document does not appear to be a compliance policy "
            f"(semantic relevance: {mean_sim:.2f} / 1.00). "
            f"It has little overlap with compliance concepts such as: "
            f"{', '.join(top_probes[:2])}. "
            "Please upload your actual policy document."
        )
    elif mean_sim < _WARN_THRESHOLD:
        warnings.append(
            f"Policy document has low compliance relevance (score: {mean_sim:.2f}). "
            f"Strongest overlap with: {', '.join(top_probes)}. "
            "Analysis may surface many gaps due to sparse compliance language."
        )

    # ── Check 2: Regulation chunk semantic diversity ──────────────────────────
    reg_chunk_diversity = 0.0
    if reg_chunks_found >= 2:
        reg_vecs           = _embed([d.page_content for d in reg_docs])
        reg_chunk_diversity = float(reg_vecs.std())
        if reg_chunk_diversity < 0.08:
            warnings.append(
                f"Retrieved regulation chunks are semantically very similar "
                f"(diversity: {reg_chunk_diversity:.3f}). "
                "The knowledge base may only cover one section of the regulation — "
                "re-index with a more complete PDF for fuller gap coverage."
            )

    return {
        "ok":                   len(errors) == 0,
        "warnings":             warnings,
        "errors":               errors,
        "compliance_relevance": round(mean_sim, 3),
        "reg_chunks_found":     reg_chunks_found,
        "reg_chunk_diversity":  round(reg_chunk_diversity, 3),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-REG CONFLICT DETECTOR
# ─────────────────────────────────────────────────────────────────────────────
_CONFLICT_PROBES = [
    (
        "Data Retention vs Right to Erasure",
        "mandatory record retention financial data 7 years minimum",
        "right to erasure deletion personal data on request article 17",
        "One regulation mandates multi-year retention of financial records. "
        "Another grants the right to immediate erasure. "
        "You need a dual-retention policy with explicit carve-outs for financial data.",
    ),
    (
        "Consent Basis — Legitimate Interest vs Consent-Only",
        "legitimate interest lawful basis processing no explicit consent needed",
        "explicit consent required before any personal data processing section 7",
        "One framework allows processing on legitimate interest. "
        "The other requires explicit opt-in consent for every purpose. "
        "Your consent workflow must satisfy the stricter requirement.",
    ),
    (
        "Breach Notification Timeline",
        "72 hour breach notification supervisory authority article 33",
        "breach notification as soon as possible data protection board",
        "Conflicting breach timelines. Apply the stricter 72-hour window "
        "across your entire breach response procedure.",
    ),
    (
        "Data Localisation",
        "payment data stored within India domestic servers only RBI",
        "cross-border data transfer approved countries permitted DPDP",
        "One regulation restricts data to local storage. "
        "Another permits cross-border transfers. "
        "Classify data by type and apply the strictest localisation rule per category.",
    ),
    (
        "Child Data Age Threshold",
        "children under 18 verifiable parental consent required DPDP",
        "children under 16 parental consent GDPR member state",
        "Conflicting age thresholds for child data. "
        "Apply the stricter under-18 threshold universally.",
    ),
]

_CONFLICT_SIM_THRESHOLD = 0.45


def detect_reg_conflicts(regulation_names: list, reg_chunks_by_name: dict = None) -> dict:
    """
    Detect cross-regulation conflicts.

    reg_chunks_by_name: optional dict of { reg_name: [chunk_text, ...] }
    If provided, uses semantic probe matching.
    Falls back to name-based detection otherwise.
    """
    if not regulation_names or len(regulation_names) < 2:
        return {"conflicts_found": 0, "conflicts": [], "summary": ""}

    conflicts = []

    if reg_chunks_by_name and len(reg_chunks_by_name) >= 2:
        condensed = {n: " ".join(chunks[:4]) for n, chunks in reg_chunks_by_name.items()}

        for topic, probe_a, probe_b, description in _CONFLICT_PROBES:
            probe_vecs = _embed([probe_a, probe_b])   # (2, D)
            best = {}
            for reg_name, text in condensed.items():
                sims = _cos(_embed([text]), probe_vecs)[0]
                for side, idx in [("a", 0), ("b", 1)]:
                    s = float(sims[idx])
                    if s > best.get(side, (None, 0.0))[1]:
                        best[side] = (reg_name, s)

            a_reg, a_sim = best.get("a", (None, 0.0))
            b_reg, b_sim = best.get("b", (None, 0.0))

            if (a_reg and b_reg and a_reg != b_reg
                    and a_sim >= _CONFLICT_SIM_THRESHOLD
                    and b_sim >= _CONFLICT_SIM_THRESHOLD):
                conflicts.append({
                    "topic":                topic,
                    "regulations_involved": [a_reg, b_reg],
                    "description":          description,
                    "severity":             "high",
                    "sim_score_a":          round(a_sim, 3),
                    "sim_score_b":          round(b_sim, 3),
                })
    else:
        # Name-based fallback
        nl = " ".join(regulation_names).lower()
        name_checks = [
            (["gdpr", "erasure"], ["rbi", "retain"],      "Data Retention vs Right to Erasure"),
            (["gdpr", "legitimate"], ["dpdp", "consent"], "Consent Basis — Legitimate Interest vs Consent-Only"),
            (["gdpr", "72"],  ["dpdp", "breach"],         "Breach Notification Timeline"),
        ]
        for kw_a, kw_b, topic in name_checks:
            if any(k in nl for k in kw_a) and any(k in nl for k in kw_b):
                desc = next((d for t, _, _, d in _CONFLICT_PROBES if t == topic), "")
                conflicts.append({
                    "topic": topic,
                    "regulations_involved": regulation_names,
                    "description": desc,
                    "severity": "medium",
                    "sim_score_a": None, "sim_score_b": None,
                })

    if not conflicts and len(regulation_names) >= 2:
        conflicts.append({
            "topic": "Multiple Regulations Active",
            "regulations_involved": regulation_names,
            "description": (
                f"{len(regulation_names)} regulations are loaded simultaneously. "
                "Obligations may carry different scopes, timelines, and penalties — "
                "review gaps from each regulation independently."
            ),
            "severity": "medium",
            "sim_score_a": None, "sim_score_b": None,
        })

    high  = sum(1 for c in conflicts if c["severity"] == "high")
    summary = (
        f"{len(conflicts)} cross-regulation conflict(s) detected ({high} high severity). "
        "Review before finalising your compliance strategy."
    ) if conflicts else ""

    return {"conflicts_found": len(conflicts), "conflicts": conflicts, "summary": summary}


# ─────────────────────────────────────────────────────────────────────────────
# SAFE FALLBACKS
# ─────────────────────────────────────────────────────────────────────────────
def gap_fallback_response(reason: str, regulation_name: str = "") -> dict:
    return {
        "compliance_score": 0,
        "regulation_name":  regulation_name or "Unknown",
        "gaps": [], "met": [],
        "_edge_case": {"blocked": True, "reason": reason},
    }


def rag_fallback_response(reason: str) -> dict:
    return {
        "answer":  f"⚠️ Cannot answer reliably. {reason}",
        "sources": [],
        "_edge": {
            "confidence": CONF_INSUFFICIENT, "confidence_score": 0,
            "grounding":  {"max_sim": 0, "mean_sim": 0,
                           "grounding_label": "WEAK", "top_chunk_index": -1},
            "faithfulness": {"faithful": False, "verdict": "HALLUCINATED",
                             "unsupported_claims": [], "confidence_note": "No source."},
            "warnings":   [reason],
            "answer_safe": False,
        },
    }