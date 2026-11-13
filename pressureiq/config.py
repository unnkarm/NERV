"""Central configuration. Every methodological constant lives here so a run is reproducible
from (raw data + Config)."""
from dataclasses import dataclass, field, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR, PROCESSED_DIR, OUTPUT_DIR = ROOT / "data/raw", ROOT / "data/processed", ROOT / "data/outputs"

# Canonical column -> accepted source names (lower-case, spaces -> underscores).
# If your CSV headers differ, add them here; nothing else needs to change.
BBB_ALIASES = {
    "match_id": ["matchid", "id", "match"],
    "innings": ["inning", "innings_no", "inns", "innings_number"],
    "over": ["over_number", "over_no", "overs", "overid"],
    "ball": ["ball_number", "ball_in_over", "delivery"],
    "batter": ["batsman", "striker", "batter_name", "batsman_name", "striker_name"],
    "non_striker": ["non_striker_name", "nonstriker"],
    "bowler": ["bowler_name"],
    "batting_team": ["bat_team", "batting_team_name", "team_batting"],
    "bowling_team": ["bowl_team", "bowling_team_name", "team_bowling"],
    "runs_batter": ["batter_runs", "runs_off_bat", "batsman_runs", "runs_scored", "bat_runs"],
    "runs_extras": ["extras", "extra_runs", "extras_runs"],
    "runs_total": ["total_runs", "total_run", "runs"],
    "extras_type": ["extra_type", "extras_kind", "extra_kind"],
    "wicket_type": ["dismissal_kind", "dismissal_type", "wicket_kind", "wicket"],
    "player_out": ["player_dismissed", "dismissed_player", "dismissed", "dismissed_player_id"],
    "is_super_over": ["super_over"],
}
MATCH_ALIASES = {
    "match_id": ["matchid", "id"],
    "competition_id": ["competition", "tournament_id", "league_id", "comp_id"],
    "season": ["year"],
    "date": ["match_date", "start_date"],
    "team1": ["team_1", "home_team"],
    "team2": ["team_2", "away_team"],
    "winner": ["match_winner", "winning_team"],
    "format": ["match_format", "match_type"],
    "target_runs": ["target", "revised_target", "dl_target"],
    "target_overs": ["revised_overs", "dl_overs", "second_inn_revised_overs"],
    "overs": ["match_overs", "scheduled_overs", "first_inn_revised_overs"],
}
COMP_ALIASES = {"competition_id": ["id", "competition", "comp_id"], "name": ["competition_name", "tournament"],
                "format": ["match_format"]}
PLAYER_ALIASES = {"player_id": ["id"], "player_name": ["name", "player"]}

# Dismissals credited to the bowler (laws of cricket) / not a dismissal at all.
BOWLER_CREDITED = {"bowled", "caught", "caught and bowled", "lbw", "stumped", "hit wicket"}
NOT_A_WICKET = {"retired hurt", "retired not out"}


@dataclass
class Config:
    # ---- match state
    max_balls: int = 120
    pp_frac: float = 0.30            # powerplay = first 30% of the innings (6/20 overs)
    death_frac: float = 0.80         # death   = last 20% (overs 17-20)
    collapse_window: int = 12        # deliveries used for "recent" run rate / wickets
    # ---- pressure model
    weights: dict = field(default_factory=lambda: {"R": 0.30, "W": 0.20, "T": 0.15, "C": 0.35})
    k_chase_rate: float = 2.0        # steepness of RRR / reference-rate logistic
    k_first_rate: float = 1.0        # steepness of phase-rate shortfall logistic
    k_required: float = 4.0          # steepness for par / projected-total ratio terms
    closeness_sigma: float = 0.6     # width of the "match in the balance" bell (log-ratio units)
    min_innings_calib: int = 30      # fewer complete innings in a format -> use pooled calibration
    pressure_q: float = 0.75         # top 25% of a format's pressure scores = "high pressure"
    low_q: float = 0.50              # below the median = "low pressure"
    # ---- evaluation / reliability
    k_phase: float = 60.0            # shrinkage (balls) of a player's phase baseline to league
    min_pressure_balls: int = 60
    min_pressure_matches: int = 8
    min_estimation_balls: int = 10   # players used to estimate the population (prior) spread
    consistency_min_balls: int = 3   # a match counts for consistency if >= this many pressure balls
    consistency_prior_matches: float = 6.0
    rating_weights: dict = field(default_factory=lambda: {"abs": 0.55, "res": 0.30, "cons": 0.15})
    rating_center: float = 50.0
    rating_scale: float = 15.0
    ci_z: float = 1.645              # 90% interval
    seed: int = 42

    def to_dict(self):
        return asdict(self)


# Alternative pressure definitions used for ranking-stability tests.
ALT_WEIGHTS = {
    "equal": {"R": .25, "W": .25, "T": .25, "C": .25},
    "rate_heavy": {"R": .60, "W": .10, "T": .10, "C": .20},
    "closeness_heavy": {"R": .10, "W": .20, "T": .10, "C": .60},
    "no_time": {"R": .40, "W": .30, "T": .00, "C": .30},
}
