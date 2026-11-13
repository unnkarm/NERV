"""Stage 05: sample-size adjustment, shrinkage, uncertainty, consistency and the composite rating.

Shrinkage is empirical-Bayes (normal-normal):
    adj = lambda * obs + (1 - lambda) * mu,     lambda = tau2 / (tau2 + se^2)
Because se^2 = sigma^2 / n, this equals the lambda = n / (n + k) form with k = sigma^2 / tau2,
except that k is *estimated from the data* (DerSimonian-Laird) instead of hand-tuned.
"""
import numpy as np
import pandas as pd
from .config import Config


def eb_shrink(obs, se, floor=1e-9):
    obs, se = np.asarray(obs, float), np.asarray(se, float)
    ok = np.isfinite(obs) & np.isfinite(se) & (se > 0)
    w0 = 1 / se[ok] ** 2
    mu0 = np.sum(w0 * obs[ok]) / np.sum(w0)
    Q = np.sum(w0 * (obs[ok] - mu0) ** 2)
    c = np.sum(w0) - np.sum(w0 ** 2) / np.sum(w0)
    tau2 = max((Q - (ok.sum() - 1)) / c, floor) if ok.sum() > 2 and c > 0 else floor
    w = 1 / (se[ok] ** 2 + tau2)
    mu = np.sum(w * obs[ok]) / np.sum(w)
    lam = np.where(np.isfinite(se) & (se > 0), tau2 / (tau2 + np.where(np.isfinite(se), se, 1e9) ** 2), 0.0)
    obs_f = np.where(np.isfinite(obs), obs, mu)
    post = mu + lam * (obs_f - mu)
    sd = np.sqrt(np.where(lam > 0, lam * np.where(np.isfinite(se), se, 1e9) ** 2, tau2))
    return post, sd, lam, mu, tau2


def player_stats(m, sigma_ball):
    """Per-player observed value-over-expectation per 100 balls with a match-clustered standard error."""
    g = m.groupby("player")
    n, v = g["n"].sum(), g["voe"].sum()
    mean = v / n
    res = m["voe"] - m["player"].map(mean) * m["n"]
    ss = (res ** 2).groupby(m["player"]).sum()
    G = g["match_id"].nunique()
    se_cl = np.sqrt(ss * G / (G - 1).clip(lower=1)) / n
    se_iid = sigma_ball / np.sqrt(n)
    se = np.maximum(se_cl.where(G > 1, 0.0), se_iid)
    return pd.DataFrame({"n": n, "matches": G, "per100": 100 * mean, "se100": 100 * se})


def consistency(pm_pressure, cfg: Config):
    m = pm_pressure[pm_pressure.n >= cfg.consistency_min_balls]
    g = m.assign(hit=(m.voe > 0).astype(int)).groupby("player")
    c = pd.DataFrame({"c_matches": g.size(), "c_hits": g.hit.sum(),
                      "voe_sd_match": g.apply(lambda x: (100 * x.voe / x.n).std())})
    p0 = c.c_hits.sum() / max(c.c_matches.sum(), 1)
    k = cfg.consistency_prior_matches
    c["cons_raw"] = c.c_hits / c.c_matches
    c["cons_adj"] = (c.c_hits + k * p0) / (c.c_matches + k)
    c["cons_se"] = np.sqrt(c.cons_adj * (1 - c.cons_adj) / (c.c_matches + k))
    return c


def rate_group(pm, cfg: Config, sigma_ball, min_balls, min_matches):
    """Ratings for one population (a single format, or 'ALL' formats pooled)."""
    P = player_stats(pm[pm.is_pressure], sigma_ball).add_prefix("p_")
    N = player_stats(pm[~pm.is_pressure], sigma_ball).add_prefix("q_")
    C = consistency(pm[pm.is_pressure], cfg)
    t = P.join(N, how="left").join(C, how="left")
    t["abs_obs"] = t.p_per100
    t["abs_se"] = t.p_se100
    t["res_obs"] = t.p_per100 - t.q_per100.fillna(t.p_per100)
    t["res_se"] = np.sqrt(t.p_se100 ** 2 + t.q_se100.fillna(1e3) ** 2)
    est = t.p_n >= cfg.min_estimation_balls
    out = {}
    for key in ["abs", "res"]:
        post = pd.Series(np.nan, index=t.index); sd = post.copy(); lam = post.copy()
        pp, ss, ll, mu, tau2 = eb_shrink(t.loc[est, f"{key}_obs"], t.loc[est, f"{key}_se"])
        post[est], sd[est], lam[est] = pp, ss, ll
        t[f"{key}_adj"], t[f"{key}_sd"], t[f"{key}_lambda"] = post, sd, lam
        out[f"{key}_prior_mean"], out[f"{key}_tau"] = float(mu), float(np.sqrt(tau2))
    t["cons_adj"] = t.cons_adj.fillna(t.cons_adj.mean())
    t["cons_sd"] = t.cons_se.fillna(t.cons_se.mean())
    t["eligible"] = (t.p_n >= min_balls) & (t.p_matches >= min_matches) & est
    e = t[t.eligible]
    rw = cfg.rating_weights
    comps = {"abs": ("abs_adj", "abs_sd", "abs_obs"), "res": ("res_adj", "res_sd", "res_obs"), "cons": ("cons_adj", "cons_sd", "cons_raw")}
    z, comp, var = {}, 0.0, 0.0
    scale = cfg.rating_scale
    for k, (col, sdc, obs) in comps.items():
        # Scale = spread of the UNSHRUNK estimates, so weak evidence compresses ratings toward 50 instead of
        # being re-inflated by rank-normalisation.
        ref = max(float(e[obs].std()), 1e-6)
        z[k] = (t[col] - e[col].mean()) / ref
        comp = comp + rw[k] * z[k]
        var = var + (rw[k] * t[sdc] / ref) ** 2
    t["rating"] = cfg.rating_center + scale * comp
    for k in comps:
        t[f"contrib_{k}"] = scale * rw[k] * z[k]       # rating = center + sum of contributions
    half = cfg.ci_z * scale * np.sqrt(var)
    t["rating_lo"], t["rating_hi"] = t.rating - half, t.rating + half
    t["confidence"] = t.abs_lambda
    t["confidence_tier"] = pd.cut(t.confidence, [-1, 0.4, 0.7, 2], labels=["low", "medium", "high"])
    t.loc[~t.eligible, ["rating", "rating_lo", "rating_hi"]] = np.nan
    t["rank"] = t.rating.rank(ascending=False, method="min")
    t.attrs.update(out)
    return t.reset_index()
