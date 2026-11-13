"""Stage 04-06: batting and bowling evaluation, baselines, rating tables.

Per-delivery value over expectation (VOE), in runs:
  batter : (runs - E[runs]) - (dismissed - E[dismissed]) * wicket_value
  bowler : -(runs_charged - E[runs]) + (credited_wicket - E[wicket]) * wicket_value
E[.] is the league rate for the same (format, innings phase, pressure level) cell, so a player is
compared with everybody else facing the same situation -- format and phase are controlled for, and
nobody is rewarded for playing more matches (all ratings are per-ball and shrunk by sample size).
"""
import numpy as np
import pandas as pd
from .config import Config
from .pressure import pressure_score, assign_levels
from .reliability import rate_group

KEYS = ["format", "phase", "level"]


def prepare_rows(states, side, cfg: Config, weights=None, q=None, naive=False):
    """One row per delivery relevant to `side`, with pressure level and VOE. Cacheable."""
    weights, q = weights or cfg.weights, q if q is not None else cfg.pressure_q
    d = states[states.state_valid].copy()
    if naive:   # strawman: "death overs == pressure"
        d["pressure"] = np.where(d.phase == "death", 100.0, 0.0)
        d["level"] = np.where(d.phase == "death", "high", "low")
    else:
        d["pressure"] = pressure_score(d, weights, side)
        d["level"], _ = assign_levels(d["pressure"], d["format"], q, cfg.low_q)
    d["is_pressure"] = d["level"] == "high"
    if side == "bat":
        d = d[~d.is_wide].copy()
        d["player"], d["team"] = d.batter, d.get("batting_team", "")
        d["unit"], d["val_runs"], d["wkt"] = 1.0, d.runs_batter, d.batter_out.astype(float)
        d["bnd"], d["dot"] = d.runs_batter.isin([4, 6]), d.runs_batter == 0
    else:
        d["player"], d["team"] = d.bowler, d.get("bowling_team", "")
        d["unit"], d["val_runs"], d["wkt"] = d.legal.astype(float), d.bowler_runs, d.bowler_wicket.astype(float)
        d["bnd"], d["dot"] = d.runs_batter.isin([4, 6]), d.legal & (d.bowler_runs == 0)
    cell = d.groupby(KEYS).agg(r=("val_runs", "sum"), w=("wkt", "sum"), u=("unit", "sum"))
    cell = pd.DataFrame({"e_run": cell.r / cell.u, "e_wkt": cell.w / cell.u})
    d = d.join(cell, on=KEYS)
    exp_r, exp_w = d.e_run * d.unit, d.e_wkt * d.unit
    if side == "bat":
        d["voe"] = (d.val_runs - exp_r) - (d.wkt - exp_w) * d.wicket_value
    else:
        d["voe"] = -(d.val_runs - exp_r) + (d.wkt - exp_w) * d.wicket_value
    return d


def _player_match(rows):
    g = ["player", "format", "match_id", "season", "is_pressure"]
    return rows.groupby(g, observed=True).agg(
        n=("unit", "sum"), runs=("val_runs", "sum"), wkt=("wkt", "sum"), bnd=("bnd", "sum"), dot=("dot", "sum"),
        voe=("voe", "sum"), pr=("pressure", "sum"), rows=("pressure", "size")).reset_index()


def _phase_baseline(rows, cfg):
    """Phase-adjusted baseline: a player's own non-pressure scoring rate per (format, phase), shrunk toward the
    league rate and re-weighted to the phase mix of that player's pressure balls."""
    g = rows.groupby(["player", "format", "phase", "is_pressure"], observed=True).agg(n=("unit", "sum"), runs=("val_runs", "sum")).reset_index()
    lg = g[~g.is_pressure].groupby(["format", "phase"]).agg(n=("n", "sum"), runs=("runs", "sum"))
    lg = (lg.runs / lg.n).rename("lr")
    p = g[g.is_pressure].set_index(["player", "format", "phase"])[["n"]].rename(columns={"n": "n_p"})
    q = g[~g.is_pressure].set_index(["player", "format", "phase"])[["n", "runs"]].rename(columns={"n": "n_q", "runs": "r_q"})
    t = p.join(q, how="left").fillna(0).reset_index().join(lg, on=["format", "phase"])
    t["rate"] = (t.r_q + cfg.k_phase * t.lr) / (t.n_q + cfg.k_phase)
    t["exp_runs"] = t.rate * t.n_p
    return t


def _summary(pm, pbase, side):
    rows = []
    for grp in list(pm.format.unique()) + ["ALL"]:
        m = pm if grp == "ALL" else pm[pm.format == grp]
        b = pbase if grp == "ALL" else pbase[pbase.format == grp]
        agg = lambda x, s: x.groupby("player").agg(**{f"n{s}": ("n", "sum"), f"matches{s}": ("match_id", "nunique"),
                                                      f"runs{s}": ("runs", "sum"), f"wkt{s}": ("wkt", "sum"),
                                                      f"bnd{s}": ("bnd", "sum"), f"dot{s}": ("dot", "sum")})
        P, N = agg(m[m.is_pressure], "_p"), agg(m[~m.is_pressure], "_q")
        s = P.join(N, how="outer").fillna(0)
        s = s.join(m[m.is_pressure].groupby("player").agg(avg_pressure=("pr", "sum"), _rows=("rows", "sum")).pipe(
            lambda x: pd.DataFrame({"avg_pressure": x.avg_pressure / x._rows})), how="left")
        s = s.join(b.groupby("player").agg(exp_runs=("exp_runs", "sum"), exp_n=("n_p", "sum")), how="left")
        for x in ["p", "q"]:
            n = s[f"n_{x}"].replace(0, np.nan)
            if side == "bat":
                s[f"sr_{x}"] = 100 * s[f"runs_{x}"] / n
            else:
                s[f"econ_{x}"] = 6 * s[f"runs_{x}"] / n
            s[f"bnd_pct_{x}"] = 100 * s[f"bnd_{x}"] / n
            s[f"dot_pct_{x}"] = 100 * s[f"dot_{x}"] / n
            s[f"wkt_per100_{x}"] = 100 * s[f"wkt_{x}"] / n
        if side == "bat":
            s["sr_baseline_adj"] = 100 * s.exp_runs / s.exp_n
            s["delta_sr_raw"], s["delta_sr_adj"] = s.sr_p - s.sr_q, s.sr_p - s.sr_baseline_adj
            s["delta_dismissal_per100"] = s.wkt_per100_p - s.wkt_per100_q
        else:
            s["econ_baseline_adj"] = 6 * s.exp_runs / s.exp_n
            s["delta_econ_raw"], s["delta_econ_adj"] = s.econ_q - s.econ_p, s.econ_baseline_adj - s.econ_p
            s["delta_wkt_per100"] = s.wkt_per100_p - s.wkt_per100_q
        s["delta_bnd_pct"] = s.bnd_pct_p - s.bnd_pct_q
        s["delta_dot_pct"] = s.dot_pct_p - s.dot_pct_q
        s = s.drop(columns=["exp_runs", "exp_n"]).reset_index()
        s.insert(1, "group", grp)
        rows.append(s)
    return pd.concat(rows, ignore_index=True)


def aggregate_side(rows, side, cfg: Config, filters=None):
    """Aggregate prepared rows into player ratings. `filters` (format/season/team lists) restrict the evidence;
    league expectations and pressure thresholds stay fixed at their full-data values."""
    for k, v in (filters or {}).items():
        if v:
            rows = rows[rows[k].isin(v)]
    sigma = float(rows.voe.std())
    pm = _player_match(rows)
    pbase = _phase_baseline(rows, cfg)
    summ = _summary(pm, pbase, side)
    rt = []
    min_b = cfg.min_pressure_balls
    for grp in list(pm.format.unique()) + ["ALL"]:
        sub = pm if grp == "ALL" else pm[pm.format == grp]
        if sub.is_pressure.sum() == 0:
            continue
        r = rate_group(sub, cfg, sigma, min_b, cfg.min_pressure_matches)
        r.insert(1, "group", grp)
        rt.append(r)
    rating_columns = [
        "player", "group", "p_n", "p_matches", "abs_adj", "res_adj", "rating", "rating_lo", "rating_hi",
        "confidence", "confidence_tier", "rank", "contrib_abs", "contrib_res", "contrib_cons",
    ]
    ratings = pd.concat(rt, ignore_index=True) if rt else pd.DataFrame(columns=rating_columns)
    ratings = summ.merge(ratings.drop(columns=[c for c in ratings.columns if c in summ.columns and c not in ("player", "group")],
                                      errors="ignore"), on=["player", "group"], how="left")
    ratings["side"] = side
    phase = rows.groupby(["player", "format", "phase", "is_pressure"], observed=True).agg(
        n=("unit", "sum"), runs=("val_runs", "sum"), wkt=("wkt", "sum"), voe=("voe", "sum")).reset_index()
    return {"ratings": ratings, "player_match": pm, "phase": phase, "sigma_ball": sigma}


def run_side(states, side, cfg: Config, weights=None, q=None, naive=False, filters=None):
    rows = prepare_rows(states, side, cfg, weights, q, naive)
    res = aggregate_side(rows, side, cfg, filters)
    res["rows"] = rows
    return res
