# NERV — Cricket Performance Intelligence

A descriptive, reproducible evaluation engine. Rankings are *outputs* of the pipeline; nothing is hand-assigned.

```
pip install -r requirements.txt
# 1. put the six CSVs in data/raw/   (ball_by_ball.csv + matches.csv are what the engine needs)
python -m pressureiq.pipeline                 # -> data/processed/, data/outputs/ (ratings, sensitivity, PDF)
streamlit run dashboard.py                    # interactive dashboard
pytest                                        # 11 tests
python -m pressureiq.pipeline --demo          # SYNTHETIC data end-to-end smoke run (data/demo/), not real cricket
```

## Layout
```
pressureiq/
  config.py     all constants + column aliases          ingest.py    load, schema map, clean, data-quality audit
  state.py      match state before every delivery       pressure.py  calibration + 0-100 pressure scores
  evaluate.py   batting/bowling VOE, baselines          reliability.py  shrinkage, CIs, consistency, rating
  pipeline.py   orchestration + sensitivity analysis    report.py    Plotly + PDF report
  synth.py      synthetic data with planted traits (demo/tests)
dashboard.py    Streamlit app (DuckDB over parquet)     tests/       pytest suite
```

## IMPORTANT: your CSV headers
I did not have your data files, so column names are matched through aliases in `pressureiq/config.py`
(`BBB_ALIASES`, `MATCH_ALIASES`). `ingest.py` fails loudly and lists the columns it found if a required one
(`match_id, innings, over, batter, bowler, runs_batter`) is missing. Strongly recommended columns:
`runs_extras, runs_total, extras_type (wides/noballs/byes/legbyes), wicket_type, player_out, batting_team,
bowling_team`. The `overid`, `striker_name`, `bowler_name`, `runs`, `extras1_runs` / `extras_2_runs`, and
`dismissed_player_id` fields are also supported; extra-run components are summed and dismissal IDs are resolved
through `competition_team_player_mapping.csv`. Delivery order = file order within (match, innings). Format
(IPL / T20I) is read from `matches.format`, else `competitions.csv` via `competition_id` (or `comp_id`). Check
`data/outputs/data_quality.csv` first.

## Method in brief
1. **State** (`state.py`): score, wickets, legal balls, run rates, target, RRR from earlier deliveries only. Wides/no-balls
   are not legal balls; byes/leg-byes are team extras; run-outs/retirements are not bowler wickets; super overs excluded;
   `target_runs` / `target_overs` columns (D/L) honoured.
2. **Pressure** (`pressure.py`): `P = 100*(wR*R + wW*W + wT*T + wC*C)`. An expected-remaining-runs surface
   `E[b,w]=A(b)*B(w)` is fitted per format on complete first innings, so 9 wkts/60 balls and 2 wkts/12 balls are compared
   on resources. Chase C = leverage x "match in the balance"; first-innings R/C = shortfall vs phase norm / par.
   High pressure = top 25% of a format's scores (configurable, tested at 60-90%).
3. **Evaluation** (`evaluate.py`): per-delivery value over expectation vs the league rate in the same (format, phase,
   pressure level) cell, wickets priced from the calibrated resource curve. Descriptive columns (pressure SR/economy,
   boundary/dot %, dismissal rate, phase-adjusted baseline deltas) are exported alongside.
4. **Reliability** (`reliability.py`): empirical-Bayes shrinkage, `lambda = tau^2/(tau^2+se^2)` (the `n/(n+k)` form
   with k estimated from the data), match-clustered standard errors, consistency (hit rate, shrunk).
   Rating = 50 + 15 x weighted z of absolute value (.55), resilience vs own normal (.30), consistency (.15).
   Contributions are exported, so `rating = 50 + contrib_abs + contrib_res + contrib_cons` exactly.
5. **Validation**: `sensitivity.csv` re-rates under alternative weights, thresholds and a naive "death overs" definition
   (Spearman, top-N overlap). On synthetic data with planted traits the engine recovers them (tests).

## Honest limitations
- Weights (pressure and rating) are documented methodological choices: tested for stability, not uniquely identified.
- Batting position/role and opposition strength are not controlled. Rating intervals are approximate (components assumed independent).
- On synthetic data the pipeline is validated; on your real data you still need to read the sensitivity table and data-quality report.
- **On ML**: none is used; the engine is descriptive, per your brief. `scikit-learn` is listed only if you later add e.g. a win-probability model.
