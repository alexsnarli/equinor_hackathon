from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR / "src"))

from volve_optimizer.data import (  # noqa: E402
    PRODUCERS,
    build_model_frame,
    choke_support,
    historical_actions,
    historical_production,
    load_daily_data,
    resolve_data_path,
)
from volve_optimizer.feedback import save_scenario_feedback  # noqa: E402
from volve_optimizer.model import train_model  # noqa: E402
from volve_optimizer.optimizer import (  # noqa: E402
    optimize_scenarios,
    simulate_scenario,
)


st.set_page_config(
    page_title="Volve Production Planner",
    page_icon="V",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      :root {
        --ink: #0f172a;
        --muted: #5b6778;
        --paper: #ffffff;
        --panel: #ffffff;
        --accent: #047857;
        --accent-dark: #065f46;
        --line: #e4e8ed;
        --soft: #f6f8fa;
        --soft-green: #f0f7f4;
      }
      .stApp { background: var(--paper); color: var(--ink); }
      [data-testid="stToolbar"] { visibility: hidden; height: 0; }
      [data-testid="stSidebar"] { background: var(--paper); }
      .block-container { max-width: 1180px; padding: 2rem 2rem 3rem; }
      html, body, [class*="css"] { font-size: 16px; }
      h1, h2, h3, h4 { color: var(--ink); letter-spacing: -.02em; }
      .app-header { margin-bottom: 2rem; }
      .app-header h1 { margin: 0; font-size: 1.75rem; line-height: 1.2; }
      .app-header p { margin: .4rem 0 0; color: var(--muted); font-size: 1rem; }
      .step-title { margin: 0; font-size: 1.2rem; font-weight: 750; }
      .step-copy { color: var(--muted); margin: .3rem 0 1rem; font-size: .9rem; }
      .well-card-title { line-height: 1.15; }
      .well-card-title strong { color: var(--ink); font-size: 1rem; }
      .well-card-title span { color: var(--muted); display: block; font-size: .72rem; margin-top: .2rem; }
      .well-state { color: var(--muted); font-size: .75rem; margin-top: -.35rem; }
      .overview-title { margin: 0 0 .85rem; font-size: 1.1rem; font-weight: 750; }
      .info-icon {
        align-items: center;
        border: 1px solid #9aa5b1;
        border-radius: 50%;
        color: var(--muted);
        cursor: help;
        display: inline-flex;
        font-size: .68rem;
        font-weight: 750;
        height: 1.15rem;
        justify-content: center;
        margin-left: .3rem;
        vertical-align: .08rem;
        width: 1.15rem;
      }
      .measured-card {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: .75rem;
      }
      .current-plan-card {
        background: var(--soft);
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 1.2rem 1.3rem;
        margin-bottom: .75rem;
      }
      .preview-label { color: #386451; font-size: .76rem; font-weight: 750; text-transform: uppercase; letter-spacing: .04em; }
      .measured-card h2, .current-plan-card h2 { margin: .3rem 0 .9rem; font-size: 1.15rem; }
      .output-grid { display: grid; grid-template-columns: 1fr 1.2fr .85fr; gap: 1rem; }
      .key-value span { color: var(--muted); display: block; font-size: .76rem; font-weight: 650; }
      .key-value strong { color: var(--ink); display: block; font-size: 1.45rem; line-height: 1.2; margin-top: .25rem; font-variant-numeric: tabular-nums; }
      .key-value small { color: var(--muted); font-size: .78rem; }
      .source-note { color: var(--muted); font-size: .78rem; margin-top: .75rem; }
      .plan-name { color: var(--ink); font-size: .95rem; font-weight: 700; margin: .15rem 0; }
      .plan-settings { color: var(--muted); font-size: .73rem; line-height: 1.35; }
      .plan-uplift { color: #285742; font-size: .8rem; font-weight: 650; margin-top: .35rem; }
      .plan-note { color: var(--muted); font-size: .75rem; line-height: 1.35; margin-top: .3rem; }
      .feedback-title { color: var(--ink); font-size: .92rem; font-weight: 700; margin-top: .5rem; }
      .limitations {
        background: var(--soft);
        border: 1px solid var(--line);
        border-radius: 12px;
        color: var(--muted);
        font-size: .82rem;
        line-height: 1.5;
        margin-top: 1.5rem;
        padding: 1rem 1.2rem;
      }
      .limitations strong { color: var(--ink); }
      .limitations ul { margin: .55rem 0 0; padding-left: 1.15rem; }
      .limitations li { margin: .25rem 0; }
      div[data-testid="stMetric"] {
        background: var(--soft);
        border: 0;
        border-radius: 10px;
        padding: 1rem;
      }
      div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: var(--ink);
        font-variant-numeric: tabular-nums;
      }
      div[data-testid="stMetric"] [data-testid="stMetricValue"] p {
        font-size: clamp(1.2rem, 2vw, 1.55rem);
        white-space: nowrap;
      }
      div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
        color: var(--muted) !important;
      }
      div[data-testid="stMetric"] [data-testid="stMetricLabel"] p {
        color: var(--muted) !important;
        font-size: .78rem;
        font-weight: 650;
      }
      div[data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--panel);
        border-color: var(--line);
        border-radius: 10px;
      }
      .confidence-note {
        border-left: 3px solid #b7791f;
        background: #fffbeb;
        padding: .7rem .9rem;
        color: #684c13;
      }
      .stButton > button[kind="primary"] {
        background: var(--accent);
        border-color: var(--accent);
        font-weight: 650;
      }
      .stButton > button[kind="primary"]:hover {
        background: var(--accent-dark);
        border-color: var(--accent-dark);
      }
      .stButton > button { border-radius: 8px; min-height: 2.8rem; }
      [data-testid="stMain"] [data-testid="stWidgetLabel"] p {
        color: #334155 !important;
        font-weight: 600;
      }
      [data-testid="stCaptionContainer"] { color: var(--muted); }
      [data-testid="stExpander"] { border-color: var(--line); background: var(--panel); }
      [data-testid="stDataFrame"] { font-variant-numeric: tabular-nums; }
      @media (max-width: 800px) {
        .block-container { padding: 1rem; }
        .output-grid { grid-template-columns: 1fr 1fr; gap: .75rem; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_source(path: str) -> pd.DataFrame:
    return load_daily_data(path)


@st.cache_resource(show_spinner="Training the Volve scenario model…")
def build_model(path: str):
    daily = load_daily_data(path)
    frame = build_model_frame(daily)
    return train_model(frame)


def output_delta(candidate: float, baseline: float, unit: str = "Sm³/day") -> str:
    return f"{candidate - baseline:+,.0f} {unit}"


def load_plan(actions: dict[str, dict[str, float | bool]], source: str) -> None:
    st.session_state["plan_chokes"] = {
        label: int(round(float(actions[label]["choke"]))) for label in PRODUCERS
    }
    for label in PRODUCERS:
        st.session_state[f"enabled_{label}"] = bool(actions[label]["on"])
        st.session_state[f"choke_{label}"] = int(round(float(actions[label]["choke"])))
    st.session_state["preview_source"] = source
    st.session_state["feedback_confirmation"] = None
    st.session_state["selected_well"] = next(
        (label for label in PRODUCERS if actions[label]["on"]),
        next(iter(PRODUCERS)),
    )


def mark_custom_plan(choke_label: str | None = None) -> None:
    if choke_label is not None:
        st.session_state["plan_chokes"][choke_label] = int(
            st.session_state[f"choke_{choke_label}"]
        )
    st.session_state["preview_source"] = "Custom settings"
    st.session_state["feedback_confirmation"] = None


def record_feedback(
    verdict: str,
    state_date: pd.Timestamp,
    scenario_name: str,
    actions: dict[str, dict[str, float | bool]],
    totals: dict[str, float],
) -> None:
    save_scenario_feedback(
        APP_DIR / "data" / "model_feedback.csv",
        state_date=state_date.date().isoformat(),
        scenario_name=scenario_name,
        verdict=verdict,
        comment=st.session_state.get("feedback_comment", ""),
        totals=totals,
        actions=actions,
    )
    st.session_state["feedback_confirmation"] = f"Saved: {verdict}"
    st.session_state["feedback_comment"] = ""


try:
    workbook_path = resolve_data_path()
except FileNotFoundError as error:
    st.error(str(error))
    st.code(
        "export VOLVE_DATA_PATH='/absolute/path/Volve production data.xlsx'\n"
        "streamlit run ask-volve/production-optimizer/app.py"
    )
    st.stop()

daily = load_source(str(workbook_path))
bundle = build_model(str(workbook_path))
support = choke_support(daily)

min_date = daily["DATEPRD"].min().date()
max_date = (daily["DATEPRD"].max() - pd.Timedelta(days=1)).date()
preferred_date = pd.Timestamp("2016-04-20").date()
default_date = min(max(preferred_date, min_date), max_date)

state_defaults = {
    "demo_as_of": default_date,
    "feedback_comment": "",
}
for key, value in state_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

as_of = pd.Timestamp(st.session_state["demo_as_of"])
constraints = {
    "max_gas": None,
    "max_water": None,
    "max_liquid": None,
    "max_pressure": None,
}

st.markdown(
    """
    <div class="app-header">
      <h1>Tomorrow's production plan</h1>
      <p>Compare yesterday's production with tomorrow's options, then test your own settings.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

baseline_actions = historical_actions(daily, as_of)
_, baseline_totals = simulate_scenario(bundle, daily, as_of, baseline_actions)
measured_totals = historical_production(daily, as_of)

if (
    st.session_state.get("planner_ui_version") != 3
    or st.session_state.get("well_state_date") != as_of.date()
):
    load_plan(baseline_actions, "Keeping yesterday's settings")
    st.session_state["well_state_date"] = as_of.date()
    st.session_state["planner_ui_version"] = 3

recommendation_signature = (
    as_of.date(),
    st.session_state["planner_ui_version"],
)
if st.session_state.get("recommendation_signature") != recommendation_signature:
    with st.spinner("Calculating ranked plans…"):
        st.session_state["recommendations"] = optimize_scenarios(
            bundle, daily, as_of, support, constraints, top_n=2
        )
    st.session_state["recommendation_signature"] = recommendation_signature

actions = {
    label: {
        "on": bool(st.session_state[f"enabled_{label}"]),
        "choke": float(st.session_state["plan_chokes"][label])
        if st.session_state[f"enabled_{label}"]
        else 0.0,
    }
    for label in PRODUCERS
}
scenario_wells, scenario_totals = simulate_scenario(bundle, daily, as_of, actions)
recommendations = st.session_state["recommendations"]

overview_left, overview_right = st.columns([.9, 1.1], gap="large")

with overview_left:
    st.markdown('<div class="overview-title">Starting point</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="measured-card">
          <div class="preview-label">Yesterday · measured</div>
          <h2>{as_of.strftime('%-d %B %Y')}</h2>
          <div class="output-grid">
            <div class="key-value">
              <span>OIL</span>
              <strong>{measured_totals['oil']:,.0f}</strong>
              <small>Sm³/day</small>
            </div>
            <div class="key-value">
              <span>GAS</span>
              <strong>{measured_totals['gas']:,.0f}</strong>
              <small>Sm³/day</small>
            </div>
            <div class="key-value">
              <span>WATER</span>
              <strong>{measured_totals['water']:,.0f}</strong>
              <small>Sm³/day</small>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="current-plan-card">
          <div class="preview-label">Tomorrow · forecast preview</div>
          <h2>{st.session_state.get('preview_source', 'Custom settings')}</h2>
          <div class="output-grid">
            <div class="key-value">
              <span>OIL</span>
              <strong>{scenario_totals['oil']:,.0f}</strong>
              <small>{output_delta(scenario_totals['oil'], measured_totals['oil'])} vs yesterday</small>
            </div>
            <div class="key-value">
              <span>GAS</span>
              <strong>{scenario_totals['gas']:,.0f}</strong>
              <small>{output_delta(scenario_totals['gas'], measured_totals['gas'])} vs yesterday</small>
            </div>
            <div class="key-value">
              <span>WATER</span>
              <strong>{scenario_totals['water']:,.0f}</strong>
              <small>{output_delta(scenario_totals['water'], measured_totals['water'])} vs yesterday</small>
            </div>
          </div>
          <div class="source-note">Predicted from the measured well state and the selected configuration.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.button(
        "Reset to yesterday's settings",
        on_click=load_plan,
        args=(baseline_actions, "Keeping yesterday's settings"),
    )
    st.markdown('<div class="feedback-title">Does this forecast look plausible?</div>', unsafe_allow_html=True)
    feedback_verdict = st.radio(
        "Model feedback",
        options=[
            "Possible",
            "Not possible",
            "Plausible",
        ],
        horizontal=True,
        label_visibility="collapsed",
    )
    st.text_input(
        "Optional note",
        key="feedback_comment",
        placeholder="What should the model learn from this?",
    )
    st.button(
        "Save feedback",
        on_click=record_feedback,
        args=(
            feedback_verdict,
            as_of,
            st.session_state.get("preview_source", "Custom settings"),
            actions,
            scenario_totals,
        ),
    )
    if st.session_state.get("feedback_confirmation"):
        st.success(st.session_state["feedback_confirmation"])

with overview_right:
    st.markdown(
        '<div class="overview-title">Suggested strategies '
        '<span class="info-icon" tabindex="0" aria-label="Strategy selection method" '
        'title="Maximum output uses gas to choose among plans within 1% of the highest predicted oil. The other card explores a one-day rest for the leading pressure-recovery candidate.">i</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    if recommendations.empty:
        st.warning("No producing plan could be calculated for this date.")
    else:
        for index, recommendation in recommendations.iterrows():
            strategy_name = str(recommendation["strategy_name"])
            strategy_note = str(recommendation["strategy_note"])
            note_markup = (
                f'<div class="plan-note">{strategy_note}</div>' if strategy_note else ""
            )
            with st.container(border=True):
                plan_copy, plan_action = st.columns([3, 1], vertical_alignment="center")
                plan_copy.markdown(
                    f"""
                    <div class="plan-name">{strategy_name}</div>
                    <div class="plan-uplift">Oil {recommendation['oil']:,.0f} ({output_delta(recommendation['oil'], baseline_totals['oil'])}) · Gas {recommendation['gas']:,.0f} ({output_delta(recommendation['gas'], baseline_totals['gas'])})</div>
                    <div class="plan-settings">{recommendation['configuration']}</div>
                    {note_markup}
                    """,
                    unsafe_allow_html=True,
                )
                plan_action.button(
                    "Preview",
                    key=f"preview_plan_{index}",
                    on_click=load_plan,
                    args=(recommendation["actions"], strategy_name),
                    width="stretch",
                )

st.divider()
st.markdown(
    '<div class="step-title">Test a custom setting</div>'
    '<div class="step-copy">Adjust any plan. The preview above updates immediately.</div>',
    unsafe_allow_html=True,
)
editor_left, editor_right = st.columns([1.2, 1], gap="large")

with editor_left:
    well_columns = st.columns(3)
    for index, (label, tag) in enumerate(PRODUCERS.items()):
        with well_columns[index % 3]:
            with st.container(border=True):
                heading, control = st.columns([1.4, 1], vertical_alignment="center")
                heading.markdown(
                    f'<div class="well-card-title"><strong>{label}</strong><span>{tag}</span></div>',
                    unsafe_allow_html=True,
                )
                enabled = control.toggle(
                    f"{label} online",
                    key=f"enabled_{label}",
                    label_visibility="collapsed",
                    on_change=mark_custom_plan,
                )
                st.markdown(
                    f'<div class="well-state">{"Producing" if enabled else "Off"}</div>',
                    unsafe_allow_html=True,
                )

with editor_right:
    selected_well = st.selectbox(
        "Adjust choke for",
        options=list(PRODUCERS),
        key="selected_well",
        format_func=lambda label: f"{label}  ·  {PRODUCERS[label]}",
    )
    selected_support = support.loc[selected_well] if selected_well in support.index else None
    if f"choke_{selected_well}" not in st.session_state:
        st.session_state[f"choke_{selected_well}"] = st.session_state["plan_chokes"][
            selected_well
        ]
    st.slider(
        "Choke opening",
        min_value=0,
        max_value=100,
        step=1,
        format="%d%%",
        key=f"choke_{selected_well}",
        disabled=not st.session_state[f"enabled_{selected_well}"],
        on_change=mark_custom_plan,
        args=(selected_well,),
    )
    if not st.session_state[f"enabled_{selected_well}"]:
        st.caption("Turn this well on to adjust its choke.")
    elif selected_support is not None:
        st.caption(f"Observed range: {selected_support['p05']:.0f}–{selected_support['p95']:.0f}%")
    st.caption("Preview only. No settings are sent to control systems.")

st.caption("Prototype prediction from historical Volve data; not validated for operations.")

with st.expander("Data settings", expanded=False):
    st.date_input(
        "Current state date",
        min_value=min_date,
        max_value=max_date,
        key="demo_as_of",
        help="Predictions use information available through this date.",
    )

with st.expander("Technical details", expanded=False):
    st.markdown(
        '<div class="confidence-note"><strong>Prototype boundary:</strong> These are historically '
        "calibrated estimates, not a validated physical network simulation. The model uses each "
        "well's pressure history, shared platform pressure summaries and the complete proposed "
        "well configuration.</div>",
        unsafe_allow_html=True,
    )
    display_wells = scenario_wells[
        ["well", "status", "choke_pct", "oil", "gas", "water"]
    ].rename(
        columns={
            "well": "Well",
            "status": "State",
            "choke_pct": "Choke %",
            "oil": "Oil",
            "gas": "Gas",
            "water": "Water",
        }
    )
    st.dataframe(
        display_wells.style.format(
            {
                "Choke %": "{:.0f}",
                "Oil": "{:,.0f}",
                "Gas": "{:,.0f}",
                "Water": "{:,.0f}",
            },
            na_rep="—",
        ),
        width="stretch",
        hide_index=True,
    )

st.markdown(
    """
    <div class="limitations">
      <strong>Prototype constraints — use for scenario discussion, not operating instructions</strong>
      <ul>
        <li>Compare relative options; absolute forecasts are not validated against a physical network model.</li>
        <li>Shared pressure is represented by historical well and platform pressure proxies, not separator or manifold hydraulics.</li>
        <li>The rest plan estimates production during the shut-in day; it does not estimate pressure recovery or the later rebound.</li>
        <li>Daily data cannot represent explosive starts, ramping or minute-to-minute control response.</li>
        <li>No downstream capacity, safety or operating constraints are applied in this demo.</li>
        <li>Settings outside historical operating patterns are less reliable and always require engineer review.</li>
      </ul>
    </div>
    """,
    unsafe_allow_html=True,
)
