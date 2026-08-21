from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from volve_optimizer.data import (  # noqa: E402
    PRODUCERS,
    build_model_frame,
    choke_support,
    historical_actions,
    load_daily_data,
    resolve_data_path,
)
from volve_optimizer.model import chronological_split, train_model  # noqa: E402
from volve_optimizer.optimizer import optimize_scenarios, simulate_scenario  # noqa: E402


class VolveOptimizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = resolve_data_path(os.getenv("VOLVE_DATA_PATH"))
        cls.daily = load_daily_data(cls.path)
        cls.frame = build_model_frame(cls.daily)
        cls.bundle = train_model(cls.frame, final_estimators=40)
        cls.as_of = pd.Timestamp("2016-04-20")

    def test_all_producers_are_mapped(self):
        self.assertEqual(set(self.daily["app_well"].dropna().unique()), set(PRODUCERS))

    def test_chronological_split_has_no_per_well_leakage(self):
        split = chronological_split(self.frame)
        for _, group in self.frame.assign(split=split).groupby("well_label"):
            train = group[group["split"].eq("train")]
            validation = group[group["split"].eq("validation")]
            test = group[group["split"].eq("test")]
            self.assertLess(train["DATEPRD"].max(), validation["DATEPRD"].min())
            self.assertLess(validation["DATEPRD"].max(), test["DATEPRD"].min())

    def test_all_off_scenario_is_zero(self):
        actions = {label: {"on": False, "choke": 0.0} for label in PRODUCERS}
        _, totals = simulate_scenario(self.bundle, self.daily, self.as_of, actions)
        self.assertEqual(totals["oil"], 0)
        self.assertEqual(totals["gas"], 0)
        self.assertEqual(totals["water"], 0)

    def test_active_scenario_predictions_are_nonnegative(self):
        actions = historical_actions(self.daily, self.as_of)
        per_well, totals = simulate_scenario(self.bundle, self.daily, self.as_of, actions)
        self.assertTrue((per_well[["oil", "gas", "water"]] >= 0).all().all())
        self.assertGreater(totals["oil"], 0)

    def test_optimizer_respects_tight_gas_constraint(self):
        recommendations = optimize_scenarios(
            self.bundle,
            self.daily,
            self.as_of,
            choke_support(self.daily),
            {
                "max_gas": 250_000.0,
                "max_water": None,
                "max_liquid": None,
                "max_pressure": None,
            },
            top_n=3,
        )
        self.assertFalse(recommendations.empty)
        self.assertTrue((recommendations["gas"] <= 250_000.0).all())


if __name__ == "__main__":
    unittest.main()
