"""
OpenAI Agents SDK orchestration surface.

The Agents SDK acts as the runtime that:
  * receives the patient's medication list,
  * calls the deterministic lookup tool, and
  * runs a guarded "report writer" agent that summarises the result
    while being strictly constrained to NOT give medical advice.

If the OpenAI key is missing or the SDK is unavailable, we fall back to the
deterministic offline report so the demo never breaks.
"""

from __future__ import annotations

import os
from typing import Any

from app.crew.agents import (
    DISCLAIMER,
    SafetyReport,
    build_report_offline,
)

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def _has_key() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


async def run_agent(medications: list[str]) -> SafetyReport:
    """
    Run the Drug Interaction Checker agent pipeline.

    Strategy:
      1. Always compute the deterministic, offline interaction lookup first.
      2. If an OpenAI key + Agents SDK are available, use a guarded agent to
         produce a friendlier narrative from that structured data.
      3. Otherwise return the offline report (still fully valid).
    """
    # Fast, reliable base result from the knowledge base.
    base = build_report_offline(medications)

    if not _has_key():
        return base

    try:
        from agents import Agent, Runner, function_tool  # type: ignore
        from pydantic import BaseModel
    except Exception:
        # Agents SDK not installed -> offline path
        return base

    # Expose the structured lookup to the agent as a tool.
    @function_tool
    def interaction_lookup(meds: str) -> str:
        """Look up dangerous drug interactions for a comma-separated med list."""
        med_list = [m.strip() for m in meds.split(",") if m.strip()]
        r = build_report_offline(med_list)
        return r.model_dump_json()

    class Narrative(BaseModel):
        headline: str
        plain_language_summary: str
        next_steps: list[str]

    writer = Agent(
        name="Safety Report Writer",
        instructions=(
            "You turn a structured drug-interaction result into a calm, "
            "plain-language summary for a patient. You MUST NOT give medical "
            "advice, prescribe, or tell the patient to start/stop/change any "
            "medication. You MUST include this disclaimer verbatim: "
            f"{DISCLAIMER}"
        ),
        model=MODEL,
        tools=[interaction_lookup],
        output_type=Narrative,
    )

    prompt = (
        f"Medication list: {', '.join(medications)}. "
        "Use the interaction_lookup tool and produce a short patient summary."
    )
    try:
        result = await Runner.run(writer, prompt)
        narrative = result.final_output_as(Narrative)
        # Merge the LLM narrative into our structured report, keeping the
        # verified findings from the deterministic lookup.
        base.recommendations = narrative.next_steps or base.recommendations
        # ensure disclaimer always present
        base.disclaimer = DISCLAIMER
        return base
    except Exception:
        # Any LLM error -> safe offline result
        return base


__all__ = ["run_agent", "SafetyReport"]
