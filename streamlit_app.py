"""
Streamlit frontend for the Drug Interaction Checker Agent.

Talks directly to the existing agent pipeline (app.agents_sdk.run_agent)
-- no FastAPI server required. The same CrewAI + OpenAI Agents SDK logic runs
behind the scenes.

Run locally:
    streamlit run streamlit_app.py

Deploy: push repo to GitHub, then New app on
https://streamlit.io/cloud -> point at streamlit_app.py.
Set OPENAI_API_KEY in the app's Secrets (or your local .env).
"""

from __future__ import annotations

import os
import sys
import asyncio
from pathlib import Path

# Make sure the repo root (where the "app" package lives) is importable,
# even if Streamlit Cloud runs this file from a different working directory.
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st  # noqa: E402

from app.crew.agents import DISCLAIMER  # noqa: E402
from app.agents_sdk import run_agent  # noqa: E402

# --- Streamlit page config ---
st.set_page_config(
    page_title="Drug Interaction Checker Agent",
    page_icon="💊",
    layout="wide",
)

# --- Disclaimer banner (always visible) ---
st.markdown(
    """
    <div style="background:#422006;border:1px solid #f59e0b;color:#fde68a;
                padding:0.85rem 1rem;border-radius:8px;font-size:0.85rem;margin-bottom:1.2rem;">
    ⚠️ <strong>Not medical advice.</strong> This is an automated demonstration.
    It does not replace a doctor or pharmacist. Always confirm with a qualified
    healthcare professional before changing any medication.
    </div>
    """,
    unsafe_allow_html=True,
)

st.title("💊 Drug Interaction Checker Agent")
st.caption("Multi-agent safety report · OpenAI Agents SDK + CrewAI")

# --- Input ---
st.markdown("### Enter the patient's medications")
meds_text = st.text_area(
    "Comma-separated list",
    placeholder="e.g. warfarin, aspirin, lisinopril, spironolactone",
    height=90,
)

run = st.button("Run Safety Check", type="primary", use_container_width=True)


def _run_sync(meds: list[str]) -> dict:
    """Call the async agent pipeline from Streamlit's sync context."""
    try:
        # If a loop is already running (some threaded setups), reuse it.
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Run in a dedicated thread so we don't block the running loop.
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(lambda: asyncio.run(run_agent(meds)))
            report = future.result()
    else:
        report = asyncio.run(run_agent(meds))

    # Return a plain dict so the UI code can use .get(...)
    return report.model_dump()


if run and meds_text.strip():
    medications = [m.strip() for m in meds_text.split(",") if m.strip()]
    with st.spinner(
        "Analyzing with the agent crew (Researcher → Analyst → Writer → Reviewer)…"
    ):
        report = _run_sync(medications)

    overall = (report.get("overall_risk") or "low").lower()
    color = {
        "low": "#22c55e",
        "moderate": "#f59e0b",
        "high": "#ef4444",
        "contraindicated": "#fca5a5",
    }.get(overall, "#94a3b8")

    st.markdown("## Safety Report")
    st.markdown(
        f"**Overall risk:** "
        f"<span style='background:{color}22;color:{color};padding:0.2rem 0.7rem;"
        f"border-radius:999px;font-weight:700;'>{overall.upper()}</span>",
        unsafe_allow_html=True,
    )

    found = report.get("patient_medications", [])
    if found:
        st.markdown(
            "**Medications checked:** " + ", ".join(f"`{m}`" for m in found)
        )
    unknown = report.get("unrecognised_medications", [])
    if unknown:
        st.error("Not recognised: " + ", ".join(unknown))

    findings = report.get("findings", [])
    if findings:
        st.markdown("### Flagged interactions")
        for f in findings:
            sev = f.get("severity", "unknown")
            sev_color = {
                "contraindicated": "#ef4444",
                "high": "#ef4444",
                "moderate": "#f59e0b",
                "low": "#22c55e",
            }.get(sev, "#94a3b8")
            with st.container():
                st.markdown(
                    f"<span style='background:{sev_color}22;color:{sev_color};"
                    f"padding:0.15rem 0.6rem;border-radius:999px;font-weight:700;"
                    f"font-size:0.8rem;'>{sev.upper()}</span> "
                    f"**{f.get('drug_a')} + {f.get('drug_b')}**",
                    unsafe_allow_html=True,
                )
                st.markdown(f"**Mechanism:** {f.get('mechanism','')}")
                st.markdown(f"**Management:** {f.get('management','')}")
                st.divider()
    else:
        st.success(
            "No known dangerous interactions found for the recognised medications."
        )

    recs = report.get("recommendations", [])
    if recs:
        st.markdown("### Recommendations")
        for r in recs:
            st.markdown(f"- {r}")

    st.markdown("---")
    st.caption(report.get("disclaimer", DISCLAIMER))

elif run and not meds_text.strip():
    st.warning("Please enter at least one medication.")


# --- Show config status in the sidebar ---
with st.sidebar:
    st.markdown("### Status")
    st.write("OpenAI key set:", bool(os.getenv("OPENAI_API_KEY")))
    st.write("Model:", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    st.caption(DISCLAIMER)
    st.write("OpenAI key set:", bool(os.getenv("OPENAI_API_KEY")))
    st.write("Model:", os.getenv("OPENAI_MODEL", "gpt- -mini"))
    st.caption(DISCLAIMER)
