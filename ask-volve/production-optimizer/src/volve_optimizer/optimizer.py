from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np
import pandas as pd

from .data import (
    FEATURE_COLUMNS,
    PRODUCERS,
    apply_actions_to_template,
    historical_actions,
    state_feature_template,
)
from .model import ModelBundle, predict


def simulate_scenario(
    bundle: ModelBundle,
    daily: pd.DataFrame,
    as_of_date: pd.Timestamp,
    actions: dict[str, dict[str, float | bool]],
) -> tuple[pd.DataFrame, dict[str, float]]:
    feature_rows: list[dict[str, Any]] = []
    active_labels: list[str] = []
    unavailable: list[str] = []

    for label in PRODUCERS:
        if not actions[label]["on"]:
            continue
        template = state_feature_template(daily, as_of_date, label)
        if template is None:
            unavailable.append(label)
            continue
        feature_rows.append(apply_actions_to_template(template, label, actions))
        active_labels.append(label)

    predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if feature_rows:
        means, deviations = predict(bundle, pd.DataFrame(feature_rows, columns=FEATURE_COLUMNS))
        predictions = {
            label: (means[index], deviations[index]) for index, label in enumerate(active_labels)
        }

    rows = []
    for label, well in PRODUCERS.items():
        action = actions[label]
        if label in unavailable:
            rows.append(
                {
                    "well": label,
                    "tag": well,
                    "status": "Unavailable",
                    "choke_pct": action["choke"],
                    "oil": 0.0,
                    "gas": 0.0,
                    "water": 0.0,
                    "whp": np.nan,
                    "oil_uncertainty": np.nan,
                    "gas_uncertainty": np.nan,
                    "water_uncertainty": np.nan,
                }
            )
        elif action["on"]:
            mean, deviation = predictions[label]
            rows.append(
                {
                    "well": label,
                    "tag": well,
                    "status": "On",
                    "choke_pct": float(action["choke"]),
                    "oil": mean[0],
                    "gas": mean[1],
                    "water": mean[2],
                    "whp": mean[3],
                    "oil_uncertainty": deviation[0],
                    "gas_uncertainty": deviation[1],
                    "water_uncertainty": deviation[2],
                }
            )
        else:
            rows.append(
                {
                    "well": label,
                    "tag": well,
                    "status": "Off",
                    "choke_pct": 0.0,
                    "oil": 0.0,
                    "gas": 0.0,
                    "water": 0.0,
                    "whp": np.nan,
                    "oil_uncertainty": 0.0,
                    "gas_uncertainty": 0.0,
                    "water_uncertainty": 0.0,
                }
            )

    per_well = pd.DataFrame(rows)
    totals = {
        "oil": float(per_well["oil"].sum()),
        "gas": float(per_well["gas"].sum()),
        "water": float(per_well["water"].sum()),
        "liquid": float((per_well["oil"] + per_well["water"]).sum()),
        "max_pressure": float(per_well["whp"].max()) if per_well["whp"].notna().any() else 0.0,
    }
    return per_well, totals


def _candidate_levels(
    label: str,
    support: pd.DataFrame,
    template_available: bool,
) -> list[float]:
    if not template_available or label not in support.index:
        return [0.0]
    row = support.loc[label]
    levels = [0.0, float(row["p25"]), float(row["median"]), float(row["p75"])]
    return sorted({round(float(np.clip(level, 0, 100)), 1) for level in levels})


def _is_feasible(row: pd.Series, constraints: dict[str, float | None]) -> bool:
    checks = [
        constraints.get("max_gas") is None or row["gas"] <= constraints["max_gas"],
        constraints.get("max_water") is None or row["water"] <= constraints["max_water"],
        constraints.get("max_liquid") is None or row["liquid"] <= constraints["max_liquid"],
        constraints.get("max_pressure") is None
        or row["max_pressure"] <= constraints["max_pressure"],
    ]
    return bool(all(checks))


def _select_with_oil_tolerance(
    candidates: pd.DataFrame, oil_tolerance: float = 0.01
) -> pd.Series | None:
    if candidates.empty:
        return None
    maximum_oil = float(candidates["oil"].max())
    oil_tier = candidates[candidates["oil"].ge(maximum_oil * (1 - oil_tolerance))]
    return oil_tier.sort_values(["gas", "oil"], ascending=[False, False]).iloc[0]


def _pressure_percentile(values: pd.Series, current: float) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty or pd.isna(current):
        return None
    return float(numeric.le(current).mean())


def _rest_candidate(
    daily: pd.DataFrame, as_of_date: pd.Timestamp
) -> tuple[str | None, int, float]:
    actions = historical_actions(daily, as_of_date)
    candidates: list[tuple[float, str, int, float]] = []
    for label, well in PRODUCERS.items():
        if not actions[label]["on"]:
            continue
        template = state_feature_template(daily, as_of_date, label)
        if template is None:
            continue
        history = daily[
            daily["NPD_WELL_BORE_NAME"].eq(well)
            & daily["DATEPRD"].le(pd.Timestamp(as_of_date).normalize())
            & daily["is_active"]
        ]
        percentiles = [
            percentile
            for percentile in (
                _pressure_percentile(
                    history["AVG_DOWNHOLE_PRESSURE"],
                    float(template["previous_downhole_pressure"]),
                ),
                _pressure_percentile(
                    history["AVG_WHP_P"], float(template["previous_whp"])
                ),
            )
            if percentile is not None
        ]
        relative_pressure = float(np.mean(percentiles)) if percentiles else 0.5
        run_days = int(template["run_days_before"])
        depletion = 1 - relative_pressure
        run_stress = min(run_days / 90, 1)
        score = 0.75 * depletion + 0.25 * run_stress
        candidates.append((score, label, run_days, relative_pressure))
    if not candidates:
        return None, 0, 0.5
    score, label, run_days, relative_pressure = max(candidates)
    return label, run_days, relative_pressure


def _uses_moderate_chokes(actions: dict[str, dict[str, Any]], support: pd.DataFrame) -> bool:
    return all(
        not action["on"]
        or label not in support.index
        or float(action["choke"]) <= float(support.loc[label, "median"]) + 0.1
        for label, action in actions.items()
    )


def optimize_scenarios(
    bundle: ModelBundle,
    daily: pd.DataFrame,
    as_of_date: pd.Timestamp,
    support: pd.DataFrame,
    constraints: dict[str, float | None],
    top_n: int = 3,
) -> pd.DataFrame:
    templates = {
        label: state_feature_template(daily, as_of_date, label) for label in PRODUCERS
    }
    levels = [
        _candidate_levels(label, support, templates[label] is not None) for label in PRODUCERS
    ]

    scenarios: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    prediction_keys: list[tuple[int, str]] = []
    for scenario_index, choices in enumerate(product(*levels)):
        actions = {
            label: {"on": choke > 0, "choke": choke}
            for label, choke in zip(PRODUCERS, choices, strict=True)
        }
        scenario = {
            "scenario": scenario_index,
            "actions": actions,
            "oil": 0.0,
            "gas": 0.0,
            "water": 0.0,
            "liquid": 0.0,
            "max_pressure": 0.0,
        }
        scenarios.append(scenario)
        for label, action in actions.items():
            if not action["on"] or templates[label] is None:
                continue
            feature_rows.append(apply_actions_to_template(templates[label], label, actions))
            prediction_keys.append((scenario_index, label))

    if feature_rows:
        means, _ = predict(bundle, pd.DataFrame(feature_rows, columns=FEATURE_COLUMNS))
        for (scenario_index, _), mean in zip(prediction_keys, means, strict=True):
            scenario = scenarios[scenario_index]
            scenario["oil"] += float(mean[0])
            scenario["gas"] += float(mean[1])
            scenario["water"] += float(mean[2])
            scenario["liquid"] += float(mean[0] + mean[2])
            scenario["max_pressure"] = max(scenario["max_pressure"], float(mean[3]))

    summary = pd.DataFrame(
        [
            {
                "scenario": scenario["scenario"],
                "configuration": " · ".join(
                    f"{label} {action['choke']:.0f}%"
                    for label, action in scenario["actions"].items()
                    if action["on"]
                )
                or "All wells off",
                "oil": scenario["oil"],
                "gas": scenario["gas"],
                "water": scenario["water"],
                "liquid": scenario["liquid"],
                "max_pressure": scenario["max_pressure"],
                "feasible": _is_feasible(pd.Series(scenario), constraints),
                "actions": scenario["actions"],
            }
            for scenario in scenarios
        ]
    )
    feasible = summary[summary["feasible"] & summary["oil"].gt(0)]
    if feasible.empty:
        return feasible

    strategies: list[pd.Series] = []

    maximum = _select_with_oil_tolerance(feasible)
    if maximum is not None:
        maximum = maximum.copy()
        maximum["strategy_id"] = "maximum_output"
        maximum["strategy_name"] = "Maximum output"
        maximum["strategy_note"] = ""
        strategies.append(maximum)

    rest_well, run_days, relative_pressure = _rest_candidate(daily, as_of_date)
    if rest_well is not None:
        rest_options = feasible[
            feasible["actions"].map(lambda actions: not actions[rest_well]["on"])
        ]
        rest = _select_with_oil_tolerance(rest_options)
        if rest is not None:
            rest = rest.copy()
            rest["strategy_id"] = "rest_recover"
            rest["strategy_name"] = f"Rest {rest_well}"
            rest["strategy_note"] = (
                f"Weakest relative pressure signal · {run_days}-day run · future rebound not modeled."
            )
            rest["rest_well"] = rest_well
            rest["rest_pressure_percentile"] = relative_pressure
            strategies.append(rest)

    moderate_options = feasible[
        feasible["actions"].map(lambda actions: _uses_moderate_chokes(actions, support))
    ]
    moderate = _select_with_oil_tolerance(moderate_options)
    if moderate is not None:
        moderate = moderate.copy()
        moderate["strategy_id"] = "moderate_chokes"
        moderate["strategy_name"] = "Moderate chokes"
        moderate["strategy_note"] = "Active wells capped at their historical median choke."
        strategies.append(moderate)

    return pd.DataFrame(strategies[:top_n]).reset_index(drop=True)


def constraint_violations(
    totals: dict[str, float], constraints: dict[str, float | None]
) -> list[str]:
    labels = {
        "max_gas": ("Gas", "gas"),
        "max_water": ("Water", "water"),
        "max_liquid": ("Liquid", "liquid"),
        "max_pressure": ("Pressure proxy", "max_pressure"),
    }
    violations = []
    for constraint, (label, total_key) in labels.items():
        limit = constraints.get(constraint)
        if limit is not None and totals[total_key] > limit:
            violations.append(f"{label}: {totals[total_key]:,.0f} exceeds {limit:,.0f}")
    return violations
