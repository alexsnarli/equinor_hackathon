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
    facility_defaults,
    historical_actions,
    load_daily_data,
    resolve_data_path,
)
from volve_optimizer.model import train_model  # noqa: E402
from volve_optimizer.optimizer import (  # noqa: E402
    constraint_violations,
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
      .current-plan-card {
        background: var(--soft);
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 1.2rem 1.3rem;
        margin-bottom: .75rem;
      }
      .preview-label { color: #386451; font-size: .76rem; font-weight: 750; text-transform: uppercase; letter-spacing: .04em; }
      .current-plan-card h2 { margin: .3rem 0 1rem; font-size: 1.2rem; }
      .key-values { display: grid; grid-template-columns: 1.35fr 1fr; gap: 1.25rem; }
      .key-value span { color: var(--muted); display: block; font-size: .76rem; font-weight: 650; }
      .key-value strong { color: var(--ink); display: block; font-size: 1.8rem; line-height: 1.2; margin-top: .25rem; font-variant-numeric: tabular-nums; }
      .key-value small { color: var(--muted); font-size: .78rem; }
      .support-output { color: var(--muted); border-top: 1px solid var(--line); font-size: .84rem; margin-top: 1rem; padding-top: .75rem; }
      .recommendation-headline {
        background: var(--soft-green);
        border: 1px solid #d7e9e1;
        border-radius: 12px;
        padding: 1rem 1.1rem;
        margin-bottom: .75rem;
      }
      .recommendation-headline span { color: #386451; font-size: .76rem; font-weight: 750; text-transform: uppercase; }
      .recommendation-headline strong { color: var(--ink); display: block; font-size: 1.5rem; margin: .25rem 0; }
      .recommendation-headline small { color: var(--muted); font-size: .82rem; }
      .plan-rank { color: var(--accent); font-size: .74rem; font-weight: 750; text-transform: uppercase; }
      .plan-name { color: var(--ink); font-size: .95rem; font-weight: 700; margin: .15rem 0; }
      .plan-settings { color: var(--muted); font-size: .73rem; line-height: 1.35; }
      .plan-uplift { color: #285742; font-size: .8rem; font-weight: 650; margin-top: .35rem; }
      .status-ok {
        color: #285742;
        font-size: .85rem;
        padding: .2rem 0;
      }
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
        .key-values { grid-template-columns: 1fr; gap: .75rem; }
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


def percent_delta(candidate: float, baseline: float) -> str:
    if baseline <= 0:
        return "n/a"
    return f"{100 * (candidate - baseline) / baseline:+.1f}%"


def estimated_daily_value(totals: dict[str, float]) -> float:
    return (
        totals["oil"] * st.session_state["value_oil_usd_sm3"]
        + totals["gas"] * st.session_state["value_gas_usd_sm3"]
        - totals["water"] * st.session_state["cost_water_usd_sm3"]
    )


def format_usd(value: float) -> str:
    sign = "-" if value < 0 else ""
    magnitude = abs(value)
    if magnitude >= 1_000_000:
        return f"{sign}${magnitude / 1_000_000:.2f}M"
    if magnitude >= 1_000:
        return f"{sign}${magnitude / 1_000:.0f}k"
    return f"{sign}${magnitude:,.0f}"


def load_plan(actions: dict[str, dict[str, float | bool]], source: str) -> None:
    for label in PRODUCERS:
        st.session_state[f"enabled_{label}"] = bool(actions[label]["on"])
        st.session_state[f"choke_{label}"] = int(round(float(actions[label]["choke"])))
    st.session_state["preview_source"] = source
    st.session_state["selected_well"] = next(
        (label for label in PRODUCERS if actions[label]["on"]),
        next(iter(PRODUCERS)),
    )


def mark_custom_plan() -> None:
    st.session_state["preview_source"] = "Custom settings"


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
defaults = facility_defaults(daily)

min_date = daily["DATEPRD"].min().date()
max_date = (daily["DATEPRD"].max() - pd.Timedelta(days=1)).date()
preferred_date = pd.Timestamp("2016-04-20").date()
default_date = min(max(preferred_date, min_date), max_date)

state_defaults = {
    "demo_as_of": default_date,
    "limit_max_gas": round(defaults["max_gas"], -3),
    "limit_max_water": round(defaults["max_water"], -2),
    "limit_max_liquid": round(defaults["max_liquid"], -2),
    "limit_use_pressure": False,
    "limit_max_pressure": round(defaults["max_pressure"], 1),
    "value_oil_usd_sm3": 500.0,
    "value_gas_usd_sm3": 0.25,
    "cost_water_usd_sm3": 5.0,
}
for key, value in state_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

as_of = pd.Timestamp(st.session_state["demo_as_of"])
constraints = {
    "max_gas": st.session_state["limit_max_gas"],
    "max_water": st.session_state["limit_max_water"],
    "max_liquid": st.session_state["limit_max_liquid"],
    "max_pressure": (
        st.session_state["limit_max_pressure"]
        if st.session_state["limit_use_pressure"]
        else None
    ),
}

st.markdown(
    """
    <div class="app-header">
      <h1>Tomorrow's production plan</h1>
      <p>Start with yesterday's configuration, compare ranked suggestions, then test your own.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

baseline_actions = historical_actions(daily, as_of)
_, baseline_totals = simulate_scenario(bundle, daily, as_of, baseline_actions)

if (
    st.session_state.get("planner_ui_version") != 2
    or st.session_state.get("well_state_date") != as_of.date()
):
    load_plan(baseline_actions, "Yesterday's configuration")
    st.session_state["well_state_date"] = as_of.date()
    st.session_state["planner_ui_version"] = 2

recommendation_signature = (
    as_of.date(),
    constraints["max_gas"],
    constraints["max_water"],
    constraints["max_liquid"],
    constraints["max_pressure"],
)
if st.session_state.get("recommendation_signature") != recommendation_signature:
    with st.spinner("Calculating ranked plans…"):
        st.session_state["recommendations"] = optimize_scenarios(
            bundle, daily, as_of, support, constraints, top_n=3
        )
    st.session_state["recommendation_signature"] = recommendation_signature

actions = {
    label: {
        "on": bool(st.session_state[f"enabled_{label}"]),
        "choke": float(st.session_state[f"choke_{label}"])
        if st.session_state[f"enabled_{label}"]
        else 0.0,
    }
    for label in PRODUCERS
}
scenario_wells, scenario_totals = simulate_scenario(bundle, daily, as_of, actions)
violations = constraint_violations(scenario_totals, constraints)
baseline_value = estimated_daily_value(baseline_totals)
scenario_value = estimated_daily_value(scenario_totals)
recommendations = st.session_state["recommendations"]

overview_left, overview_right = st.columns([.9, 1.1], gap="large")

with overview_left:
    st.markdown('<div class="overview-title">Current preview</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="current-plan-card">
          <div class="preview-label">Previewing</div>
          <h2>{st.session_state.get('preview_source', 'Custom settings')}</h2>
          <div class="key-values">
            <div class="key-value">
              <span>ESTIMATED DAILY VALUE</span>
              <strong>{format_usd(scenario_value)}</strong>
              <small>{format_usd(scenario_value - baseline_value)} vs measured yesterday</small>
            </div>
            <div class="key-value">
              <span>OIL</span>
              <strong>{scenario_totals['oil']:,.0f}</strong>
              <small>Sm³/day · {percent_delta(scenario_totals['oil'], baseline_totals['oil'])}</small>
            </div>
          </div>
          <div class="support-output">
            Gas {scenario_totals['gas']:,.0f} Sm³/day · Water {scenario_totals['water']:,.0f} Sm³/day
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if violations:
        st.error("Outside facility limits: " + "; ".join(violations))
    else:
        st.markdown('<div class="status-ok">Within the selected facility limits</div>', unsafe_allow_html=True)
    st.button(
        "Back to yesterday's configuration",
        on_click=load_plan,
        args=(baseline_actions, "Yesterday's configuration"),
    )
    st.caption("Value uses the assumptions under Advanced settings.")

with overview_right:
    st.markdown('<div class="overview-title">Ranked suggestions</div>', unsafe_allow_html=True)
    if recommendations.empty:
        st.warning("No producing plan satisfies the selected limits.")
    else:
        best = recommendations.iloc[0]
        best_value = estimated_daily_value(best.to_dict())
        st.markdown(
            f"""
            <div class="recommendation-headline">
              <span>Recommended opportunity</span>
              <strong>{format_usd(best_value - baseline_value)}/day</strong>
              <small>{percent_delta(best['oil'], baseline_totals['oil'])} oil compared with yesterday</small>
            </div>
            """,
            unsafe_allow_html=True,
        )
        plan_names = ["Recommended plan", "Alternative 1", "Alternative 2"]
        for index, recommendation in recommendations.iterrows():
            recommendation_value = estimated_daily_value(recommendation.to_dict())
            with st.container(border=True):
                plan_copy, plan_action = st.columns([3, 1], vertical_alignment="center")
                plan_copy.markdown(
                    f"""
                    <div class="plan-rank">#{index + 1}</div>
                    <div class="plan-name">{plan_names[index]}</div>
                    <div class="plan-settings">{recommendation['configuration']}</div>
                    <div class="plan-uplift">{percent_delta(recommendation['oil'], baseline_totals['oil'])} oil · {format_usd(recommendation_value - baseline_value)}/day</div>
                    """,
                    unsafe_allow_html=True,
                )
                plan_action.button(
                    "Preview",
                    key=f"preview_plan_{index}",
                    on_click=load_plan,
                    args=(recommendation["actions"], plan_names[index]),
                    width="stretch",
                )
        st.caption("Ranked by oil, then gas, then lower water—within selected limits.")

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
    st.slider(
        "Choke opening",
        min_value=0,
        max_value=100,
        step=1,
        format="%d%%",
        key=f"choke_{selected_well}",
        disabled=not st.session_state[f"enabled_{selected_well}"],
        on_change=mark_custom_plan,
    )
    if not st.session_state[f"enabled_{selected_well}"]:
        st.caption("Turn this well on to adjust its choke.")
    elif selected_support is not None:
        st.caption(f"Observed range: {selected_support['p05']:.0f}–{selected_support['p95']:.0f}%")
    st.caption("Preview only. No settings are sent to control systems.")

st.caption("Prototype prediction from historical Volve data; not validated for operations.")

with st.expander("Advanced settings", expanded=False):
    st.date_input(
        "Current state date",
        min_value=min_date,
        max_value=max_date,
        key="demo_as_of",
        help="Predictions use information available through this date.",
    )
    setting_columns = st.columns(2)
    setting_columns[0].number_input(
        "Maximum gas (Sm³/day)", min_value=0.0, step=10_000.0, key="limit_max_gas"
    )
    setting_columns[1].number_input(
        "Maximum water (Sm³/day)", min_value=0.0, step=100.0, key="limit_max_water"
    )
    setting_columns[0].number_input(
        "Maximum liquid (Sm³/day)", min_value=0.0, step=100.0, key="limit_max_liquid"
    )
    setting_columns[1].checkbox("Use WHP proxy limit", key="limit_use_pressure")
    setting_columns[1].number_input(
        "Maximum WHP proxy",
        min_value=0.0,
        step=1.0,
        key="limit_max_pressure",
        disabled=not st.session_state["limit_use_pressure"],
    )
    st.divider()
    st.markdown("**Estimated value assumptions (USD)**")
    value_columns = st.columns(3)
    value_columns[0].number_input(
        "Oil value / Sm³", min_value=0.0, step=25.0, key="value_oil_usd_sm3"
    )
    value_columns[1].number_input(
        "Gas value / Sm³", min_value=0.0, step=0.05, key="value_gas_usd_sm3"
    )
    value_columns[2].number_input(
        "Water handling / Sm³", min_value=0.0, step=1.0, key="cost_water_usd_sm3"
    )

with st.expander("Technical details", expanded=False):
    st.markdown(
        '<div class="confidence-note"><strong>Prototype boundary:</strong> These are historically '
        "calibrated estimates, not a validated physical network simulation.</div>",
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
