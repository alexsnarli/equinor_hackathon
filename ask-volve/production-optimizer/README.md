# Volve Production Scenario Advisor

Local hackathon prototype for comparing proposed Volve well configurations and finding
historically supported alternatives under facility constraints.

## What it does

- Uses all six wellbores with production history and maps them to `W1`–`W6`.
- Trains on daily production data with chronological train, validation and test partitions.
- Predicts next-day 24-hour oil, gas, water and wellhead-pressure proxy values.
- Lets engineers toggle wells and choose continuous choke settings.
- Compares a proposal with the "do nothing" configuration from the selected date.
- Searches discrete, historically supported choke levels and returns the three best feasible
  configurations: maximize oil, then gas, then minimize water.

This is an advisory data model, not a physical network simulator or automatic control system.

## Run locally

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r ask-volve/production-optimizer/requirements.txt
export VOLVE_DATA_PATH="/absolute/path/Volve production data.xlsx"
streamlit run ask-volve/production-optimizer/app.py
```

If the workbook remains at `~/Downloads/Volve production data.xlsx`, the environment variable is
optional.

## Run tests

```bash
source .venv/bin/activate
python -m unittest discover -s ask-volve/production-optimizer/tests -v
```

## SME inputs to update

The app currently offers historical 95th-percentile defaults and an optional maximum wellhead
pressure proxy. Confirm these items before presenting the constraint logic as physically valid:

Use [SME_INTERVIEW.md](SME_INTERVIEW.md) as the five-minute interview script.

| Constraint | Exact tag | Unit | Limit | Hard or soft |
|---|---|---|---|---|
| Shared-line pressure | TBD | TBD | TBD | TBD |
| Gas flow | TBD | Sm³/day | TBD | TBD |
| Water flow | TBD | Sm³/day | TBD | TBD |
| Total liquid flow | TBD | Sm³/day | TBD | TBD |

Also confirm whether per-well phase volumes are measured or allocated and whether opening one well
materially changes other wells through common back pressure.

## Modeling boundaries

- A date represents the latest state known to the model; predictions are for the following day.
- Manual sliders accept 0–100%, but the UI warns outside each well's typical historical range.
- The optimizer uses each well's historical 25th, 50th and 75th-percentile choke settings plus off.
- Estimated uplift compares two model predictions from the same state. It is not proven production
  uplift until validated through an operational trial or a trusted physical simulator.
- Daily data cannot represent minute-scale startup transients.
