from __future__ import annotations

import os
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PRODUCERS = OrderedDict(
    [
        ("W1", "15/9-F-1 C"),
        ("W2", "15/9-F-11"),
        ("W3", "15/9-F-12"),
        ("W4", "15/9-F-14"),
        ("W5", "15/9-F-15 D"),
        ("W6", "15/9-F-5"),
    ]
)
WELL_TO_LABEL = {well: label for label, well in PRODUCERS.items()}

TARGET_COLUMNS = ["target_oil", "target_gas", "target_water", "target_whp"]
STATE_FEATURES = [
    "planned_choke",
    "previous_choke",
    "previous_oil_rate",
    "previous_gas_rate",
    "previous_water_rate",
    "rolling_oil_rate_7",
    "rolling_gas_rate_7",
    "rolling_water_rate_7",
    "previous_whp",
    "previous_downhole_pressure",
    "previous_wht",
    "rest_days_before",
    "run_days_before",
    "days_since_first_record",
    "cumulative_oil_before",
    "cumulative_gas_before",
    "cumulative_water_before",
    "day_of_year_sin",
    "day_of_year_cos",
]
WELL_FEATURES = [f"well_{label}" for label in PRODUCERS]
CONFIG_FEATURES = [
    feature
    for label in PRODUCERS
    for feature in (f"config_{label}_on", f"config_{label}_choke")
]
FEATURE_COLUMNS = STATE_FEATURES + WELL_FEATURES + CONFIG_FEATURES

NUMERIC_COLUMNS = [
    "ON_STREAM_HRS",
    "AVG_DOWNHOLE_PRESSURE",
    "AVG_DOWNHOLE_TEMPERATURE",
    "AVG_CHOKE_SIZE_P",
    "AVG_WHP_P",
    "AVG_WHT_P",
    "BORE_OIL_VOL",
    "BORE_GAS_VOL",
    "BORE_WAT_VOL",
]


def resolve_data_path(explicit_path: str | Path | None = None) -> Path:
    """Resolve the workbook without copying or mutating the supplied source file."""
    candidates = [
        explicit_path,
        os.getenv("VOLVE_DATA_PATH"),
        Path(__file__).resolve().parents[2] / "data" / "Volve production data.xlsx",
        Path.home() / "Downloads" / "Volve production data.xlsx",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).expanduser().is_file():
            return Path(candidate).expanduser().resolve()
    raise FileNotFoundError(
        "Could not find 'Volve production data.xlsx'. Set VOLVE_DATA_PATH or place "
        "the workbook in ask-volve/production-optimizer/data/."
    )


def load_daily_data(path: str | Path) -> pd.DataFrame:
    daily = pd.read_excel(path, sheet_name="Daily Production Data")
    missing = {
        "DATEPRD",
        "NPD_WELL_BORE_NAME",
        "FLOW_KIND",
        *NUMERIC_COLUMNS,
    } - set(daily.columns)
    if missing:
        raise ValueError(f"Workbook is missing required columns: {sorted(missing)}")

    daily = daily[
        daily["NPD_WELL_BORE_NAME"].isin(PRODUCERS.values())
        & daily["FLOW_KIND"].eq("production")
    ].copy()
    daily["DATEPRD"] = pd.to_datetime(daily["DATEPRD"], errors="coerce").dt.normalize()
    for column in NUMERIC_COLUMNS:
        daily[column] = pd.to_numeric(daily[column], errors="coerce")

    daily["app_well"] = daily["NPD_WELL_BORE_NAME"].map(WELL_TO_LABEL)
    daily["is_active"] = (
        daily["ON_STREAM_HRS"].fillna(0).gt(0)
        & (
            daily[["BORE_OIL_VOL", "BORE_GAS_VOL", "BORE_WAT_VOL"]]
            .fillna(0)
            .gt(0)
            .any(axis=1)
        )
    )
    hours = daily["ON_STREAM_HRS"].where(daily["ON_STREAM_HRS"].gt(0))
    daily["oil_rate_24h"] = (daily["BORE_OIL_VOL"].fillna(0) * 24 / hours).fillna(0)
    daily["gas_rate_24h"] = (daily["BORE_GAS_VOL"].fillna(0) * 24 / hours).fillna(0)
    daily["water_rate_24h"] = (daily["BORE_WAT_VOL"].fillna(0) * 24 / hours).fillna(0)
    return daily.sort_values(["NPD_WELL_BORE_NAME", "DATEPRD"]).reset_index(drop=True)


def _run_lengths_before(active: pd.Series) -> tuple[list[int], list[int]]:
    rest_before: list[int] = []
    run_before: list[int] = []
    rest = 0
    run = 0
    for is_active in active.fillna(False).astype(bool):
        rest_before.append(rest)
        run_before.append(run)
        if is_active:
            run += 1
            rest = 0
        else:
            rest += 1
            run = 0
    return rest_before, run_before


def _configuration_by_date(daily: pd.DataFrame) -> pd.DataFrame:
    dates = pd.Index(sorted(daily["DATEPRD"].dropna().unique()), name="DATEPRD")
    configuration = pd.DataFrame(index=dates)
    for label, well in PRODUCERS.items():
        well_rows = daily[daily["NPD_WELL_BORE_NAME"].eq(well)].set_index("DATEPRD")
        configuration[f"config_{label}_on"] = (
            well_rows["is_active"].reindex(dates, fill_value=False).astype(float)
        )
        configuration[f"config_{label}_choke"] = (
            well_rows["AVG_CHOKE_SIZE_P"].reindex(dates).fillna(0).clip(0, 100)
        )
    return configuration.reset_index()


def build_model_frame(daily: pd.DataFrame) -> pd.DataFrame:
    """Build active-day examples using only state available before each target day."""
    configuration = _configuration_by_date(daily)
    first_global_date = daily["DATEPRD"].min()
    parts: list[pd.DataFrame] = []

    for label, well in PRODUCERS.items():
        part = daily[daily["NPD_WELL_BORE_NAME"].eq(well)].copy()
        if part.empty:
            continue
        part = part.sort_values("DATEPRD").reset_index(drop=True)
        rest_before, run_before = _run_lengths_before(part["is_active"])
        part["rest_days_before"] = rest_before
        part["run_days_before"] = run_before

        part["planned_choke"] = part["AVG_CHOKE_SIZE_P"].clip(0, 100)
        part["previous_choke"] = part["AVG_CHOKE_SIZE_P"].shift(1)
        part["previous_oil_rate"] = part["oil_rate_24h"].shift(1)
        part["previous_gas_rate"] = part["gas_rate_24h"].shift(1)
        part["previous_water_rate"] = part["water_rate_24h"].shift(1)
        part["rolling_oil_rate_7"] = part["oil_rate_24h"].shift(1).rolling(7, min_periods=1).median()
        part["rolling_gas_rate_7"] = part["gas_rate_24h"].shift(1).rolling(7, min_periods=1).median()
        part["rolling_water_rate_7"] = part["water_rate_24h"].shift(1).rolling(7, min_periods=1).median()
        part["previous_whp"] = part["AVG_WHP_P"].shift(1)
        part["previous_downhole_pressure"] = part["AVG_DOWNHOLE_PRESSURE"].shift(1)
        part["previous_wht"] = part["AVG_WHT_P"].shift(1)
        part["days_since_first_record"] = (part["DATEPRD"] - part["DATEPRD"].min()).dt.days
        part["cumulative_oil_before"] = part["BORE_OIL_VOL"].fillna(0).cumsum().shift(1).fillna(0)
        part["cumulative_gas_before"] = part["BORE_GAS_VOL"].fillna(0).cumsum().shift(1).fillna(0)
        part["cumulative_water_before"] = part["BORE_WAT_VOL"].fillna(0).cumsum().shift(1).fillna(0)
        day_of_year = part["DATEPRD"].dt.dayofyear
        part["day_of_year_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
        part["day_of_year_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)

        for candidate_label in PRODUCERS:
            part[f"well_{candidate_label}"] = float(candidate_label == label)

        part = part.merge(configuration, on="DATEPRD", how="left")
        part["target_oil"] = part["oil_rate_24h"]
        part["target_gas"] = part["gas_rate_24h"]
        part["target_water"] = part["water_rate_24h"]
        part["target_whp"] = part["AVG_WHP_P"]
        part["well_label"] = label
        part["well_name"] = well
        part["global_day"] = (part["DATEPRD"] - first_global_date).dt.days
        parts.append(part[part["is_active"] & part["target_whp"].notna()])

    frame = pd.concat(parts, ignore_index=True)
    return frame[["DATEPRD", "well_label", "well_name", *FEATURE_COLUMNS, *TARGET_COLUMNS]]


def historical_actions(daily: pd.DataFrame, as_of_date: pd.Timestamp) -> dict[str, dict[str, float | bool]]:
    as_of_date = pd.Timestamp(as_of_date).normalize()
    actions: dict[str, dict[str, float | bool]] = {}
    for label, well in PRODUCERS.items():
        exact = daily[
            daily["NPD_WELL_BORE_NAME"].eq(well) & daily["DATEPRD"].eq(as_of_date)
        ]
        if exact.empty:
            actions[label] = {"on": False, "choke": 0.0}
            continue
        row = exact.iloc[-1]
        choke = pd.to_numeric(pd.Series([row["AVG_CHOKE_SIZE_P"]]), errors="coerce").iloc[0]
        actions[label] = {
            "on": bool(row["is_active"]),
            "choke": float(np.clip(0 if pd.isna(choke) else choke, 0, 100)),
        }
    return actions


def _last_numeric(history: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(history[column], errors="coerce").dropna()
    return float(values.iloc[-1]) if not values.empty else np.nan


def state_feature_template(
    daily: pd.DataFrame,
    as_of_date: pd.Timestamp,
    well_label: str,
) -> dict[str, Any] | None:
    as_of_date = pd.Timestamp(as_of_date).normalize()
    well = PRODUCERS[well_label]
    history = daily[
        daily["NPD_WELL_BORE_NAME"].eq(well) & daily["DATEPRD"].le(as_of_date)
    ].sort_values("DATEPRD")
    if history.empty:
        return None

    last = history.iloc[-1]
    rest_days = 0
    run_days = 0
    for active in history["is_active"].iloc[::-1]:
        if active and rest_days == 0:
            run_days += 1
        elif not active and run_days == 0:
            rest_days += 1
        else:
            break

    recent = history.tail(7)
    target_date = as_of_date + pd.Timedelta(days=1)
    day_of_year = target_date.dayofyear
    template: dict[str, Any] = {
        "previous_choke": _last_numeric(history, "AVG_CHOKE_SIZE_P"),
        "previous_oil_rate": float(last["oil_rate_24h"]),
        "previous_gas_rate": float(last["gas_rate_24h"]),
        "previous_water_rate": float(last["water_rate_24h"]),
        "rolling_oil_rate_7": float(recent["oil_rate_24h"].median()),
        "rolling_gas_rate_7": float(recent["gas_rate_24h"].median()),
        "rolling_water_rate_7": float(recent["water_rate_24h"].median()),
        "previous_whp": _last_numeric(history, "AVG_WHP_P"),
        "previous_downhole_pressure": _last_numeric(history, "AVG_DOWNHOLE_PRESSURE"),
        "previous_wht": _last_numeric(history, "AVG_WHT_P"),
        "rest_days_before": float(rest_days),
        "run_days_before": float(run_days),
        "days_since_first_record": float((target_date - history["DATEPRD"].min()).days),
        "cumulative_oil_before": float(history["BORE_OIL_VOL"].fillna(0).sum()),
        "cumulative_gas_before": float(history["BORE_GAS_VOL"].fillna(0).sum()),
        "cumulative_water_before": float(history["BORE_WAT_VOL"].fillna(0).sum()),
        "day_of_year_sin": float(np.sin(2 * np.pi * day_of_year / 365.25)),
        "day_of_year_cos": float(np.cos(2 * np.pi * day_of_year / 365.25)),
    }
    for label in PRODUCERS:
        template[f"well_{label}"] = float(label == well_label)
    return template


def apply_actions_to_template(
    template: dict[str, Any],
    well_label: str,
    actions: dict[str, dict[str, float | bool]],
) -> dict[str, Any]:
    row = dict(template)
    row["planned_choke"] = float(actions[well_label]["choke"])
    for label in PRODUCERS:
        action = actions[label]
        row[f"config_{label}_on"] = float(bool(action["on"]))
        row[f"config_{label}_choke"] = float(action["choke"]) if action["on"] else 0.0
    return {column: row.get(column, np.nan) for column in FEATURE_COLUMNS}


def choke_support(daily: pd.DataFrame) -> pd.DataFrame:
    active = daily[daily["is_active"] & daily["AVG_CHOKE_SIZE_P"].notna()]
    return (
        active.groupby("app_well")["AVG_CHOKE_SIZE_P"]
        .agg(
            minimum="min",
            p05=lambda values: values.quantile(0.05),
            p25=lambda values: values.quantile(0.25),
            median="median",
            p75=lambda values: values.quantile(0.75),
            p95=lambda values: values.quantile(0.95),
            maximum="max",
            active_days="size",
        )
        .reindex(PRODUCERS)
    )


def facility_defaults(daily: pd.DataFrame) -> dict[str, float]:
    totals = daily.groupby("DATEPRD").agg(
        oil=("BORE_OIL_VOL", "sum"),
        gas=("BORE_GAS_VOL", "sum"),
        water=("BORE_WAT_VOL", "sum"),
        pressure=("AVG_WHP_P", "max"),
    )
    return {
        "max_gas": float(totals["gas"].quantile(0.95)),
        "max_water": float(totals["water"].quantile(0.95)),
        "max_liquid": float((totals["oil"] + totals["water"]).quantile(0.95)),
        "max_pressure": float(totals["pressure"].quantile(0.95)),
    }
