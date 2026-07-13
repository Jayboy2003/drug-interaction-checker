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


def _resolve_token(token: str, db: dict[str, Any]) -> str | None:
    """Map a user-entered drug name/alias to a canonical drug key in the DB."""
    token = _normalize(token)
    drugs = db.get("drugs", {})
    # exact canonical match
    if token in drugs:
        return token
    # exact alias match
    for key, meta in drugs.items():
        aliases = [a.lower() for a in meta.get("aliases", [])]
        if token in aliases:
            return key
    # partial alias match (e.g. 'advil' vs 'ibuprofen')
    for key, meta in meta.get("aliases", []) if False else []:
        pass
    for key, meta in drugs.items():
        aliases = [a.lower() for a in meta.get("aliases", [])]
        for a in aliases:
            if token in a or a in token:
                return key
    return None


def lookup_interactions(medications: list[str]) -> LookupResult:
    """
    Given a patient's medication list (free text names), return the
    structured set of dangerous pairwise interactions found in the DB.
    """
    db = _load_db()
    interactions = db.get("interactions", [])

    resolved: dict[str, str] = {}  # original -> canonical
    unknown: list[str] = []
    for med in medications:
        canon = _resolve_token(med, db)
        if canon:
            resolved[_normalize(med)] = canon
        else:
            unknown.append(med.strip())

    canon_keys = list(set(resolved.values()))
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
