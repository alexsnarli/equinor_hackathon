# Volve Production Scenario Advisor

Local hackathon prototype for comparing proposed Volve well configurations and finding
historically supported alternatives that maximize oil, using gas as the tie-breaker.

## What it does

- Uses all six wellbores with production history and maps them to `W1`–`W6`.
- Trains on daily production data with chronological train, validation and test partitions.
- Shows yesterday's measured oil, gas and water separately from tomorrow's forecast.
- Predicts next-day 24-hour oil, gas, water and wellhead-pressure proxy values.
- Lets engineers toggle wells and choose continuous choke settings.
- Compares every proposal with the forecast from keeping yesterday's settings.
- Searches discrete, historically supported choke levels and returns two distinct strategies:
  maximum output and rest and recover.
- Saves engineer plausibility feedback, optional notes and complete scenario context to
  `data/model_feedback.csv`.

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

Use [SME_INTERVIEW.md](SME_INTERVIEW.md) to confirm the shared-pressure signals, interaction logic,
data quality and the meaning of each plausibility verdict before presenting the model as
operationally valid.

## Modeling boundaries

- A date represents the latest state known to the model; predictions are for the following day.
- The model receives each well's pressure history, shared platform pressure summaries and the full
  proposed configuration, allowing historical interactions between wells to influence predictions.
- Manual sliders accept 0–100%, but the UI warns outside each well's typical historical range.
- The optimizer uses each well's historical 25th, 50th and 75th-percentile choke settings plus off.
- Estimated uplift compares two model predictions from the same state. It is not proven production
  uplift until validated through an operational trial or a trusted physical simulator.
- No downstream capacity or safety constraints are applied in the hackathon demo.
- Daily data cannot represent minute-scale startup transients.

## Next iteration

- Add an expandable **Show more rest-and-recover suggestions** action that compares additional rest
  candidates and clearly separates the predicted shut-in-day cost from any unmodeled future rebound.
