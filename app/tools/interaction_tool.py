"""
Tool: Drug Interaction Lookup
--------------------------------
Queries the bundled offline knowledge base (data/interactions.json) for
dangerous combinations between a patient's medications.

This is intentionally deterministic and offline so the demo works without
any external API. In production you would swap/extend this with a live
clinical source (e.g. NIH RxNav, Lexicomp, Micromedex).
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "interactions.json"


class InteractionHit(BaseModel):
    drug_a: str
    drug_b: str
    severity: str
    mechanism: str
    management: str


class LookupResult(BaseModel):
    found_drugs: list[str] = Field(default_factory=list)
    unknown_drugs: list[str] = Field(default_factory=list)
    interactions: list[InteractionHit] = Field(default_factory=list)
    summary: str = ""


@lru_cache(maxsize=1)
def _load_db() -> dict[str, Any]:
    with open(DATA_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _normalize(name: str) -> str:
    return name.strip().lower()


def _tokenize(token: str) -> set[str]:
    """Split a drug name into comparable words, dropping punctuation/stopwords."""
    words = re.sub(r"[^a-z0-9 ]", " ", token.lower()).split()
    # drop common dosage / formulation noise so 'warfarin 5mg' matches 'warfarin'
    stop = {
        "mg", "mcg", "g", "ml", "tablet", "tablets", "cap", "caps", "capsule",
        "capsules", "oral", "tab", "sr", "er", "xr", "la", "immediate",
        "release", "extended", "generic", "brand",
    }
    return {w for w in words if w not in stop}


def _resolve_token(token: str, db: dict[str, Any]) -> list[str]:
    """
    Map a user-entered drug name/alias to one or more canonical drug keys.

    Returns a LIST (a name like 'nsaid' can match several drugs).
    Tries, in order:
      1. exact canonical key (case-insensitive)
      2.BaseTool exact alias (case-insensitive)
      3. token-set containment (e.g. 'warfarin 5mg' -> warfarin)
      4. substring / fuzzy containment on canonical name + aliases
    """
    token_n = _normalize(token)
    drugs = db.get("drugs", {})

    # 1. exact canonical key
    for key in drugs:
        if token_n == key.lower():
            return [key]

    # 2. exact alias
    for key, meta in drugs.items():
        for a in meta.get("aliases", []):
            if token_n == a.lower():
                return [key]

    # 3. token-set containment (handles 'warfarin 5mg', 'lisinopril 10 mg')
    tt = _tokenize(token_n)
    if tt:
        for key in drugs:
            if tt == _tokenize(key):
                return [key]
        for key, meta in drugs.items():
            ka = _tokenize(key)
            if tt and ka and (tt <= ka or ka <= tt):
                # one is contained in the other (ignoring dosage words)
                if tt & ka:  # at least one real word overlaps
                    return [key]
            for a in meta.get("aliases", []):
                if tt and tt <= _tokenize(a):
                    return [key]

    # 4. substring / fuzzy containment on name + aliases
    results: list[str] = []
    for key, meta in drugs.items():
        haystacks = [key.lower()] + [a.lower() for a in meta.get("aliases", [])]
        for h in haystacks:
            if not h:
                continue
            if token_n in h or h in token_n:
                if key not in results:
                    results.append(key)
    return results


def lookup_interactions(medications: list[str]) -> LookupResult:
    """
    Given a patient's medication list (free text names), return the
    structured set of dangerous pairwise interactions found in the DB.
    """
    db = _load_db()
    interactions = db.get("interactions", [])

    resolved_keys: set[str] = set()  # canonical keys recognised
    unknown: list[str] = []
    for med in medications:
        matches = _resolve_token(med, db)
        if matches:
            # a name like 'nsaid' can map to several -> keep all
            resolved_keys.update(m.lower() for m in matches)
        else:
            unknown.append(med.strip())

    canon_keys = sorted(resolved_keys)
    hits: list[InteractionHit] = []
    seen = set()
    for pair in interactions:
        a, b = _normalize(pair["pair"][0]), _normalize(pair["pair"][1])
        if a in canon_keys and b in canon_keys:
            sig = tuple(sorted((a, b)))
            if sig in seen:
                continue
            seen.add(sig)
            hits.append(
                InteractionHit(
                    drug_a=pair["pair"][0],
                    drug_b=pair["pair"][1],
                    severity=pair.get("severity", "unknown"),
                    mechanism=pair.get("mechanism", ""),
                    management=pair.get("management", ""),
                )
            )

    # Sort: most severe first
    order = {"contraindicated": 0, "high": 1, "moderate": 2, "low": 3}
    hits.sort(key=lambda h: order.get(h.severity, 9))

    summary = (
        f"Checked {len(canon_keys)} recognised medication(s); "
        f"found {len(hits)} potential interaction(s)."
    )
    if unknown:
        summary += f" {len(unknown)} medication name(s) were not recognised and should be verified manually."

    return LookupResult(
        found_drugs=sorted(canon_keys),
        unknown_drugs=unknown,
        interactions=hits,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# CrewAI-compatible tool wrapper
# ---------------------------------------------------------------------------
try:
    from crewai_tools import BaseTool

    class DrugInteractionTool(BaseTool):
        name: str = "drug_interaction_lookup"
        description: str = (
            "Look up dangerous pairwise drug interactions for a patient's "
            "medication list against a curated offline knowledge base. "
            "Input: a comma-separated list of medication names. "
            "Returns severity, mechanism, and management for each interaction."
        )

        def _run(self, medications: str) -> str:
            meds = [m.strip() for m in medications.split(",") if m.strip()]
            result = lookup_interactions(meds)
            if not result.interactions:
                return (
                    "No known dangerous interactions found in the knowledge base "
                    "for the recognised medications. " + result.summary
                )
            lines = [result.summary, ""]
            for h in result.interactions:
                lines.append(
                    f"[{h.severity.upper()}] {h.drug_a} + {h.drug_b}\n"
                    f"  Mechanism: {h.mechanism}\n"
                    f"  Management: {h.management}"
                )
            if result.unknown_drugs:
                lines.append(
                    "Unrecognised: " + ", ".join(result.unknown_drugs)
                )
            return "\n".join(lines)

    DRUG_TOOL = DrugInteractionTool()
except Exception:  # crewai not installed in this env -> skip wrapper
    DRUG_TOOL = None


if __name__ == "__main__":
    import sys

    sample = sys.argv[1:] or ["warfarin", "aspirin", "ibuprofen", "lisinopril"]
    print(lookup_interactions(sample).model_dump_json(indent=2))
