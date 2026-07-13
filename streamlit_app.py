"""
Streamlit frontend for the Drug Interaction Checker Agent.

This talks directly to the existing agent pipeline (app.agents_sdk.run_agent)
— no FastAPI server required. The same CrewAI + OpenAI Agents SDK logic runs
behind the scenes.

Run locally:
    streamlit run streamlit_app.py

Deploy: push repo to GitHub, then New app on
https://streamlit.io/cloud -> point at streamlit_app.py.
Set OPENAI_API_KEY in the app's Secrets (or your local .env).
"""

from __future__ import annotations

import os
import asyncio

import streamlit as st

from app.crew.agents import DISCLAIMER

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

if run and meds_text.strip():
    medications = [m.strip() for m in meds_text.split(",") if m.strip()]
    with st.spinner("Analyzing with the agent crew (Researcher → Analyst → Writer → Reviewer)…"):
        # run_agent is async; call it from sync Streamlit context
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
        st.markdown("**Medications checked:** " + ", ".join(f"`{m}`" for m in found))
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
        st.success("No known dangerous interactions found for the recognised medications.")

    recs = report.get("recommendations", [])
    if recs:
        st.markdown("### Recommendations")
        for r in recs:
            st.markdown(f"- {r}")

    st.markdown("---")
    st.caption(DISCLAIMER)

elif run and not meds_text.strip():
    st.warning("Please enter at least one medication.")


# --- helper to call the async agent pipeline from sync Streamlit context ---
def _run_sync(meds):
    from app.agents_sdk import run_agent

    try:
        # Reuse the running loop if one exists
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(run_agent(meds))
    except RuntimeError:
        # No running loop (fresh thread) -> create one
        return asyncio.run(run_agent(meds))


# Show config status in the sidebar
with st.sidebar:
    st.markdown("### Status")
    st.write("OpenAI key set:", bool(os.getenv("OPENAI_API_KEY")))
    st.write("Model:", os.getenv("OPENAI_MODEL", "gpt- -mini"))
    st.caption(DISCLAIMER)
