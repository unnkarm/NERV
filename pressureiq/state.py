"""Stage 02: rebuild the match state *before* every delivery.

Leakage rule: every state_* feature uses only deliveries strictly earlier in the same innings
(plus, for chases, the completed first-innings total, which is known when the chase starts).
`inn_total`, `inn_wk`, `inn_legal` describe the whole innings and are used ONLY for calibration
(fitting league reference tables) and completeness flags, never as a per-ball pressure input.
"""
import numpy as np
import pandas as pd
from .config import Config, BOWLER_CREDITED, NOT_A_WICKET


def build_states(balls, matches, cfg: Config):
    keep = [c for c in ["match_id", "format", "season", "target_runs", "target_overs", "overs"] if c in matches.columns]
    df = balls.merge(matches[keep], on="match_id", how="left")
    df["format"] = df["format"].fillna("T20")
    df["season"] = df["season"].fillna("NA")

    so = df["innings"] > 2
    if "is_super_over" in df.columns:
        so |= df["is_super_over"].astype(str).str.lower().isin(["1", "true", "yes"])
    df = df[~so].sort_values(["match_id", "innings", "row_order"], kind="stable").reset_index(drop=True)

    et, wt, po = df["extras_type"].fillna(""), df["wicket_type"].fillna(""), df["player_out"].fillna("")
    df["is_wide"] = et.str.contains("wides")
    df["is_noball"] = et.str.contains("noballs")
    df["is_bye"] = et.str.contains("byes") & ~et.str.contains("legbyes")
    df["is_legbye"] = et.str.contains("legbyes")
    df["legal"] = ~(df.is_wide | df.is_noball)
    df["team_wicket"] = (wt != "") & ~wt.isin(NOT_A_WICKET)
    df["bowler_wicket"] = wt.isin(BOWLER_CREDITED)
    df["batter_out"] = df.team_wicket & ((po == df["batter"]) | (po == ""))      # striker dismissed
    df["team_runs"] = df["runs_total"]
    # bowler-attributable runs: bat runs + wides + no-ball penalty; byes / leg-byes are team extras
    df["bowler_runs"] = df["runs_batter"] + np.where(df.is_wide | df.is_noball, df["runs_extras"], 0)
    for c in ["legal", "team_wicket", "bowler_wicket", "batter_out"]:
        df[f"_{c}"] = df[c].astype(int)

    keys = ["match_id", "innings"]
    G = lambda: df.groupby(keys, sort=False)
    df["runs_before"] = G()["team_runs"].cumsum() - df["team_runs"]
    df["wickets_before"] = G()["_team_wicket"].cumsum() - df["_team_wicket"]
    df["legal_before"] = G()["_legal"].cumsum() - df["_legal"]

    mb = pd.Series(cfg.max_balls, index=df.index, dtype=float)
    if "overs" in df.columns:
        mb = mb.where(df["overs"].isna(), df["overs"] * 6)
    if "target_overs" in df.columns:
        mb = mb.where(~((df.innings == 2) & df["target_overs"].notna()), df["target_overs"] * 6)
    df["max_balls"] = mb.astype(int)

    df["inn_total"] = G()["team_runs"].transform("sum")
    df["inn_wk"] = G()["_team_wicket"].transform("sum")
    df["inn_legal"] = G()["_legal"].transform("sum")
    t1 = df[df.innings == 1].groupby("match_id")["team_runs"].sum()
    df["target"] = np.where(df.innings == 2, df.match_id.map(t1) + 1, np.nan)
    if "target_runs" in df.columns:                                      # revised (D/L) targets
        df["target"] = np.where((df.innings == 2) & df["target_runs"].notna(), df["target_runs"], df["target"])

    df["innings_complete"] = ((df.inn_legal >= df.max_balls) | (df.inn_wk >= 10)
                              | ((df.innings == 2) & df.target.notna() & (df.inn_total >= df.target)))
    df["balls_remaining"] = (df.max_balls - df.legal_before).clip(lower=0)
    df["wickets_remaining"] = 10 - df.wickets_before
    df["over_idx"] = df.legal_before // 6
    frac = df.legal_before / df.max_balls
    df["phase"] = np.select([frac < cfg.pp_frac, frac < cfg.death_frac], ["powerplay", "middle"], "death")
    df["context"] = np.where(df.innings == 1, "first_innings", "chase")
    df["crr"] = np.where(df.legal_before > 0, df.runs_before * 6 / df.legal_before.clip(lower=1), np.nan)
    df["runs_required"] = df.target - df.runs_before
    df["rrr"] = np.where((df.innings == 2) & (df.balls_remaining > 0),
                         df.runs_required * 6 / df.balls_remaining.clip(lower=1), np.nan)

    W = cfg.collapse_window
    for col, name in [("team_runs", "r_recent"), ("_legal", "l_recent"), ("_team_wicket", "w_recent")]:
        df[name] = G()[col].transform(lambda s: s.astype(float).shift(1).rolling(W, min_periods=1).sum()).fillna(0)
    df["rr_recent"] = np.where(df.l_recent > 0, df.r_recent * 6 / df.l_recent.clip(lower=1), np.nan)

    chase_ok = (df.innings == 1) | (df.target.notna() & (df.runs_required > 0))
    df["state_valid"] = chase_ok & (df.balls_remaining > 0) & (df.wickets_before < 10)
    return df.drop(columns=[c for c in df.columns if c.startswith("_")])
