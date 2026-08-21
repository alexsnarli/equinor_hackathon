from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FEEDBACK_FIELDS = [
    "timestamp_utc",
    "state_date",
    "scenario_name",
    "verdict",
    "comment",
    "oil_sm3_day",
    "gas_sm3_day",
    "water_sm3_day",
    "actions_json",
]


def save_scenario_feedback(
    path: str | Path,
    *,
    state_date: str,
    scenario_name: str,
    verdict: str,
    comment: str,
    totals: dict[str, float],
    actions: dict[str, dict[str, Any]],
) -> None:
    """Append one engineer assessment and its complete scenario to a local CSV."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "state_date": state_date,
        "scenario_name": scenario_name,
        "verdict": verdict,
        "comment": comment.strip(),
        "oil_sm3_day": round(float(totals["oil"]), 3),
        "gas_sm3_day": round(float(totals["gas"]), 3),
        "water_sm3_day": round(float(totals["water"]), 3),
        "actions_json": json.dumps(actions, sort_keys=True),
    }
    write_header = not output_path.exists()
    with output_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FEEDBACK_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
