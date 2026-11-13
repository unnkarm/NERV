# NERV — Cricket Performance Intelligence

**A descriptive, data-driven cricket performance evaluation engine that identifies and ranks the most reliable batters and bowlers under pressure.**

NERV analyses ball-by-ball IPL and T20I data to evaluate how players perform in high-pressure match situations. Rather than relying on raw strike rates, economy rates, or reputation, it measures performance relative to contextual expectations, individual baselines, and statistical reliability.

Rankings are generated entirely by the evaluation pipeline. No player is manually assigned a rank.

---

## 1. Key Features

- **Match-State Reconstruction:** Reconstructs the match situation before every delivery.
- **Pressure Quantification:** Calculates a 0–100 pressure score using match context, resources, and chase dynamics.
- **Batting Evaluation:** Measures batting value, strike-rate performance, boundary contribution, and resilience under pressure.
- **Bowling Evaluation:** Evaluates run prevention, wicket-taking, dot-ball contribution, and pressure performance.
- **Reliability Adjustment:** Uses empirical-Bayes shrinkage, confidence intervals, and consistency measures to reduce small-sample bias.
- **Sensitivity Analysis:** Tests ranking stability under alternative pressure thresholds, scoring weights, and definitions.
- **Interactive Dashboard:** Explores player ratings, performance metrics, and ranking comparisons.
- **Automated Reporting:** Generates CSV outputs, visualisations, and a PDF report.

## 2. Project Workflow

The evaluation pipeline follows this sequence:

```text
Raw Cricket Data
       |
       v
Data Ingestion & Cleaning
       |
       v
Match-State Reconstruction
       |
       v
Pressure Score Calculation
       |
       v
Performance Evaluation
       |
       v
Reliability & Statistical Adjustment
       |
       v
Sensitivity Analysis
       |
       v
Final Player Ratings & Rankings
       |
       +------> CSV Outputs
       +------> Interactive Dashboard
       +------> PDF Report
```

## 3. Dataset

NERV uses six CSV files covering IPL and T20I competitions from 2022 onwards.

| File | Description |
|---|---|
| `competitions.csv` | Competition names, formats, and seasons |
| `teams.csv` | Team identifiers and names |
| `matches.csv` | Match results, venues, innings conditions, and targets |
| `ball_by_ball.csv` | Delivery-level batting, bowling, and match events |
| `competition_team_player_mapping.csv` | Player, team, and competition mappings |
| `playing_eleven.csv` | Player participation and playing XI information |

**Dataset overview:**
- 883 matches
- 200,999 ball-by-ball records
- 6,413 competition-team-player mappings
- 28,381 playing XI records

### Data Placement

Place the six CSV files inside:

```text
data/
└── raw/
    ├── competitions.csv
    ├── teams.csv
    ├── matches.csv
    ├── ball_by_ball.csv
    ├── competition_team_player_mapping.csv
    └── playing_eleven.csv
```

The core evaluation requires `ball_by_ball.csv` and `matches.csv`. The remaining files support competition metadata, player mapping, and participation validation.

### Schema Compatibility

The ingestion layer maps the supplied dataset's original column names to canonical internal names through aliases defined in `pressureiq/config.py`.

Examples:

| Original column | Canonical field |
|---|---|
| `overid` | `over` |
| `striker_name` | `batter` |
| `bowler_name` | `bowler` |
| `runs` | `runs_batter` |
| `extras1_runs`, `extras_2_runs` | Extra-run components |
| `dismissed_player_id` | `player_out` |

Competition format is obtained through `matches.csv` → `comp_id` → `competitions.csv` → `match_format`.

The ingestion module validates required fields and reports missing or incompatible columns rather than silently continuing.

Check `data/outputs/data_quality.csv` before interpreting rankings.

## 4. Methodology

### 4.1 Match-State Reconstruction

`state.py` reconstructs the match situation immediately before each delivery.

It tracks:
- Current score and wickets lost.
- Legal balls remaining.
- Current run rate.
- Target and required run rate.
- Innings and over progression.

Important rules:
- Wides and no-balls do not count as legal deliveries.
- Byes and leg-byes are treated as team extras.
- Run-outs and retirements are not credited as bowler wickets.
- Super overs are excluded.
- Revised D/L targets are considered when the necessary data is available.

### 4.2 Pressure Quantification

`pressure.py` calculates a pressure score between 0 and 100.

The pressure model considers:

- Run-rate requirements.
- Wickets remaining.
- Balls remaining.
- Match closeness and chase leverage.
- First-innings scoring context.

An expected-remaining-runs surface is calibrated using complete first innings:

`E(b, w) = A(b) × B(w)`

This helps compare different match situations according to their remaining batting resources.

High-pressure deliveries are identified using the top 25% of pressure scores within each format by default. Alternative thresholds are evaluated during sensitivity analysis.

### 4.3 Performance Evaluation

`evaluate.py` calculates player performance relative to league expectations in comparable format, innings-phase, and pressure-level conditions.

**Batting metrics:**
- Pressure strike rate.
- Runs and boundaries.
- Dismissal rate.
- Phase-adjusted performance.
- Value over expectation.

**Bowling metrics:**
- Pressure economy.
- Wicket-taking contribution.
- Dot-ball percentage.
- Runs conceded relative to expectations.
- Phase-adjusted bowling value.

Performance is evaluated at the delivery level rather than through aggregate statistics alone.

### 4.4 Reliability Adjustment

`reliability.py` addresses the problem of players having different numbers of opportunities under pressure.

It incorporates:

- Empirical-Bayes shrinkage.
- Match-clustered standard errors.
- Approximate confidence intervals.
- Performance consistency.
- Minimum sample-size requirements.

Shrinkage is calculated as:

`λ = τ² / (τ² + SE²)`

The final rating combines three components:

| Component | Weight |
|---|---:|
| Absolute performance under pressure | 55% |
| Resilience relative to normal performance | 30% |
| Performance consistency | 15% |

Final rating:

`Rating = 50 + 15 × Weighted Z-Score`

The component contributions are exported, allowing the final rating to be independently reconstructed.

### 4.5 Sensitivity Analysis

`pipeline.py` evaluates whether the rankings remain stable under alternative methodological choices.

It compares:
- Alternative pressure thresholds.
- Different scoring weights.
- Alternative pressure definitions.
- A conventional death-overs benchmark.

Ranking stability is assessed using:
- Spearman rank correlation.
- Top-N overlap.

This helps distinguish consistently strong performers from players whose rankings depend heavily on one particular definition of pressure.

## 5. Project Structure

```text
NERV/
│
├── pressureiq/
│   ├── config.py
│   ├── ingest.py
│   ├── state.py
│   ├── pressure.py
│   ├── evaluate.py
│   ├── reliability.py
│   ├── pipeline.py
│   ├── report.py
│   └── synth.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── outputs/
│   └── demo/
│
├── tests/
│
├── dashboard.py
├── requirements.txt
└── README.md
```

| Module | Responsibility |
|---|---|
| `config.py` | Constants, scoring weights, and column aliases |
| `ingest.py` | Data loading, schema mapping, cleaning, and quality checks |
| `state.py` | Match-state reconstruction |
| `pressure.py` | Pressure calibration and scoring |
| `evaluate.py` | Batting and bowling performance evaluation |
| `reliability.py` | Shrinkage, confidence intervals, consistency, and ratings |
| `pipeline.py` | Pipeline orchestration and sensitivity analysis |
| `report.py` | Visualisations and PDF report generation |
| `synth.py` | Synthetic data generation for testing and demonstrations |
| `dashboard.py` | Interactive Streamlit dashboard |

## 6. Installation & Execution

### Requirements

- Python 3.10+
- pip

### Installation

```bash
git clone <repository-url>
cd NERV

pip install -r requirements.txt
```

### Run the Evaluation Pipeline

After placing the six CSV files in `data/raw/`:

```bash
python -m pressureiq.pipeline
```

The pipeline generates processed datasets and evaluation outputs inside:

```text
data/
├── processed/
└── outputs/
    ├── ratings
    ├── sensitivity
    └── PDF report
```

Exact filenames are determined by the pipeline configuration.

### Launch the Dashboard

```bash
streamlit run dashboard.py
```

### Run the Synthetic Demonstration

```bash
python -m pressureiq.pipeline --demo
```

This runs an end-to-end demonstration using synthetic cricket data with deliberately planted player-performance characteristics.

**Important:** Synthetic demonstration results are not real cricket rankings and must not be presented as such.

### Run Tests

```bash
pytest
```

The project includes 11 tests covering core pipeline behaviour and synthetic-data validation.

## 7. Limitations & Considerations

NERV is designed as a descriptive statistical evaluation engine, not a predictive machine-learning model.

Current limitations include:

- Pressure and rating weights are methodological choices, not uniquely established parameters.
- Batting position, player role, and opposition strength are not explicitly controlled.
- Confidence intervals are approximate and assume independence between rating components.
- Synthetic-data validation does not establish the validity of every result on real cricket data.
- Final rankings should be interpreted alongside data-quality reports and sensitivity results.

Real-data evaluation should be considered complete only after reviewing the generated data-quality and sensitivity reports.

## 8. Machine Learning

No machine-learning model is currently used in the core evaluation pipeline.

NERV deliberately uses a transparent, reproducible statistical methodology so that every player rating can be traced to measurable performance contributions.

Scikit-learn may be used in future extensions, such as:
- Win-probability modelling.
- Predictive match-state analysis.
- Player performance forecasting.

These extensions are outside the current project scope.

---

**NERV — Cricket Performance Intelligence**

*Measuring performance when it matters most.*