from __future__ import annotations

import os
import sys
import tempfile
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
    historical_production,
    load_daily_data,
    resolve_data_path,
)
from volve_optimizer.feedback import save_scenario_feedback  # noqa: E402
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

    def test_measured_production_is_separate_from_forecast(self):
        measured = historical_production(self.daily, self.as_of)
        rows = self.daily[self.daily["DATEPRD"].eq(self.as_of)]
        self.assertAlmostEqual(measured["oil"], rows["BORE_OIL_VOL"].fillna(0).sum())
        self.assertAlmostEqual(measured["gas"], rows["BORE_GAS_VOL"].fillna(0).sum())

    def test_model_frame_includes_shared_pressure_state(self):
        columns = {
            "previous_system_mean_whp",
            "previous_system_max_whp",
            "previous_system_active_wells",
        }
        self.assertTrue(columns.issubset(self.frame.columns))
        self.assertTrue(self.frame[list(columns)].notna().any().all())

    def test_feedback_is_saved_with_scenario_context(self):
        actions = historical_actions(self.daily, self.as_of)
        _, totals = simulate_scenario(self.bundle, self.daily, self.as_of, actions)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "feedback.csv"
            save_scenario_feedback(
                path,
                state_date=self.as_of.date().isoformat(),
                scenario_name="Recommended plan",
                verdict="Plausible",
                comment="Looks reasonable",
                totals=totals,
                actions=actions,
            )
            saved = pd.read_csv(path)
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved.loc[0, "verdict"], "Plausible")
        self.assertEqual(saved.loc[0, "scenario_name"], "Recommended plan")

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
