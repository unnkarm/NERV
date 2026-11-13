"""Stage 03: continuous 0-100 pressure score for batters and bowlers.

P = 100 * (wR*R + wW*W + wT*T + wC*C), every component already normalised to [0, 1].

Calibration (data-driven, per format): an expected-remaining-runs surface
    E[b, w] = A(b) * B(w)         b = balls remaining, w = wickets lost
fitted by alternating least squares on COMPLETE first innings. It tells us what an innings with
b balls and w wickets lost typically still scores, so "9 wickets / 60 balls" and "2 wickets / 12 balls"
are compared on resources, not on raw required run rate. Its wicket factor B(w) also gives the
run value of a wicket, used later to price dismissals.
"""
import numpy as np
import pandas as pd
from .config import Config

COMP_COLS = ["E_rem", "wicket_value", "R_bat", "W_bat", "T", "C_bat", "R_bowl", "W_bowl", "C_bowl"]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _fit_one(d, cfg):
    mb = int(d.max_balls.max())
    rem = (d.inn_total - d.runs_before).to_numpy(float)
    b = d.balls_remaining.clip(0, mb).to_numpy(int)
    w = d.wickets_before.clip(0, 9).to_numpy(int)
    B = np.ones(10)
    for _ in range(10):
        num = np.bincount(b, weights=rem * B[w], minlength=mb + 1)
        den = np.bincount(b, weights=B[w] ** 2, minlength=mb + 1)
        A = np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)
        A = pd.Series(A).interpolate(limit_direction="both").to_numpy()
        num = np.bincount(w, weights=rem * A[b], minlength=10)
        den = np.bincount(w, weights=A[b] ** 2, minlength=10)
        B = np.where(den > 0, num / np.maximum(den, 1e-9), B)
        B = B / B[0]
    num = np.bincount(b, weights=rem * B[w], minlength=mb + 1)
    den = np.bincount(b, weights=B[w] ** 2, minlength=mb + 1)
    A = pd.Series(np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)).interpolate(limit_direction="both")
    A = A.rolling(5, center=True, min_periods=1).mean().to_numpy().copy()
    A[0] = 0.0
    A = np.maximum.accumulate(A)                  # more balls left -> at least as many runs to come
    B = np.maximum(np.minimum.accumulate(B), 0.02)  # more wickets lost -> no more runs to come
    Bf = np.append(B, 0.0)                         # 10 wickets down: nothing left
    per_inn = d.groupby("match_id").agg(total=("inn_total", "first"))
    par = float(per_inn.total.mean())
    leg = d[d.legal]
    phase_rr = (6 * leg.groupby("phase").team_runs.sum() / leg.groupby("phase").legal.sum()).to_dict()
    phase_sd = leg.groupby("phase").rr_recent.std().fillna(3.0).clip(lower=1.0).to_dict()
    return {"A": A, "B": Bf, "par_total": par, "sd_total": float(per_inn.total.std() or 20.0),
            "ref_rr": par * 6 / mb, "phase_rr": phase_rr, "phase_sd": phase_sd,
            "max_balls": mb, "n_innings": int(len(per_inn))}


class Calibration:
    def __init__(self, tables):
        self.tables = tables

    def get(self, fmt):
        return self.tables.get(fmt, self.tables["__global__"])

    def to_json(self):
        out = {}
        for k, t in self.tables.items():
            out[k] = {"n_innings": t["n_innings"], "par_total": round(t["par_total"], 2), "sd_total": round(t["sd_total"], 2),
                      "ref_run_rate": round(t["ref_rr"], 3), "phase_run_rate": {p: round(v, 3) for p, v in t["phase_rr"].items()},
                      "wicket_resource_B": [round(float(x), 3) for x in t["B"]],
                      "expected_remaining_runs_at_full_innings_A": round(float(t["A"][t["max_balls"]]), 1)}
        return out


def fit_calibration(states, cfg: Config):
    comp = states[(states.innings == 1) & states.innings_complete & states.state_valid]
    if comp.empty:
        raise ValueError("No complete first innings available to calibrate the pressure model.")
    tables = {"__global__": _fit_one(comp, cfg)}
    for fmt, d in comp.groupby("format"):
        if d.match_id.nunique() >= cfg.min_innings_calib:
            tables[fmt] = _fit_one(d, cfg)
    return Calibration(tables)


def compute_components(states, cal: Calibration, cfg: Config):
    out = pd.DataFrame(np.nan, index=states.index, columns=COMP_COLS)
    for fmt, d in states[states.state_valid].groupby("format"):
        p = cal.get(fmt)
        A, Bf = p["A"], p["B"]
        b = d.balls_remaining.clip(0, len(A) - 1).to_numpy(int)
        w = d.wickets_before.clip(0, 9).to_numpy(int)
        E = np.maximum(A[b] * Bf[w], 1.0)
        wv = A[b] * (Bf[w] - Bf[w + 1])                      # runs lost when the next wicket falls
        res = Bf[w] / Bf[0]                                  # batting resources still in hand (1 -> 0)
        T = (1 - b / d.max_balls.to_numpy(float)) ** 2       # time pressure, rises late in the innings
        chase = (d.innings == 2).to_numpy()
        phase_rr = d.phase.map(p["phase_rr"]).to_numpy(float)
        phase_sd = d.phase.map(p["phase_sd"]).to_numpy(float)
        rr = d.rr_recent.to_numpy(float)
        rr = np.where(np.isnan(rr), phase_rr, rr)
        hot = (rr - phase_rr) / phase_sd                     # >0: scoring faster than the phase norm
        # --- chase
        rrr = np.nan_to_num(d.rrr.to_numpy(float), nan=p["ref_rr"])
        req = np.nan_to_num(d.runs_required.to_numpy(float), nan=1.0)
        lnD = np.log(np.clip(req / E, 0.05, 20))             # >0: need more than an average innings would add
        R_c = sigmoid(cfg.k_chase_rate * (rrr / p["ref_rr"] - 1))
        lever = np.sqrt(1 - b / d.max_balls.to_numpy(float))          # each ball matters more as the innings runs out
        C_c = lever * np.exp(-lnD ** 2 / (2 * cfg.closeness_sigma ** 2))  # leverage x "match in the balance"
        # --- first innings
        R_f = 2 * np.maximum(sigmoid(-cfg.k_first_rate * hot) - 0.5, 0)
        W_f = 0.6 * (1 - res) + 0.4 * np.minimum(d.w_recent.to_numpy(float) / 3, 1)   # wickets + collapse
        need = p["par_total"] - d.runs_before.to_numpy(float)
        Dp = np.clip(need / E, 1e-3, 20)
        C_f = np.where(need <= 0, 0.0, 2 * np.maximum(sigmoid(cfg.k_required * np.log(Dp)) - 0.5, 0))
        # --- bowling-side mirror images
        proj = np.log(np.clip((d.runs_before.to_numpy(float) + E) / p["par_total"], 1e-3, 20))
        Rb_f = 2 * np.maximum(sigmoid(cfg.k_first_rate * hot) - 0.5, 0)
        Cb_f = 2 * np.maximum(sigmoid(cfg.k_required * proj) - 0.5, 0)
        o = pd.DataFrame({
            "E_rem": E, "wicket_value": wv, "T": T,
            "R_bat": np.where(chase, R_c, R_f), "W_bat": np.where(chase, 1 - res, W_f), "C_bat": np.where(chase, C_c, C_f),
            "R_bowl": np.where(chase, 1 - R_c, Rb_f), "W_bowl": res, "C_bowl": np.where(chase, C_c, Cb_f),
        }, index=d.index)
        out.loc[o.index, o.columns] = o
    return out


def pressure_score(df, weights, side):
    s = {"bat": ("R_bat", "W_bat", "C_bat"), "bowl": ("R_bowl", "W_bowl", "C_bowl")}[side]
    return 100 * (weights["R"] * df[s[0]] + weights["W"] * df[s[1]] + weights["T"] * df["T"] + weights["C"] * df[s[2]])


def assign_levels(score, fmt, q_hi, q_lo):
    s = pd.Series(score.to_numpy(float), index=score.index)
    hi = s.groupby(fmt).transform(lambda x: x.quantile(q_hi))
    lo = s.groupby(fmt).transform(lambda x: x.quantile(q_lo))
    lvl = pd.Series(np.where(s >= hi, "high", np.where(s >= lo, "mid", "low")), index=s.index)
    return lvl, hi
