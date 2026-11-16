"""Stage 07: charts (Plotly for the dashboard, Matplotlib for the static PDF) and the methodology report.
Every number in the report is read from the run's outputs, so the report cannot drift from the results."""
import io
import numpy as np
import pandas as pd


# ---------------------------------------------------------------- Plotly (dashboard)
def fig_pressure_distribution(states, side="bat"):
    import plotly.express as px
    from .pressure import pressure_score
    from .config import Config
    d = states[states.state_valid & states.W_bat.notna()]
    d = d.sample(min(60000, len(d)), random_state=0)
    d = d.assign(pressure=pressure_score(d, Config().weights, side))
    return px.histogram(d, x="pressure", color="context", facet_col="format", nbins=50, barmode="overlay",
                        opacity=.65, title=f"{side} pressure score distribution")


def fig_components(states):
    import plotly.express as px
    d = states[states.state_valid].groupby("phase")[["R_bat", "W_bat", "T", "C_bat"]].mean().reset_index().melt("phase")
    return px.bar(d, x="phase", y="value", color="variable", barmode="group", title="Mean pressure components by phase")


def fig_rank_bars(df, title):
    import plotly.graph_objects as go
    d = df.sort_values("rating").tail(25)
    f = go.Figure(go.Bar(x=d.rating, y=d.player, orientation="h",
                         error_x=dict(type="data", symmetric=False, array=d.rating_hi - d.rating, arrayminus=d.rating - d.rating_lo)))
    f.update_layout(title=title, height=620, xaxis_title="rating (50 = average, 90% interval)")
    return f


def fig_pressure_vs_normal(row, side):
    import plotly.graph_objects as go
    k = ("sr", "Strike rate") if side == "bat" else ("econ", "Economy")
    f = go.Figure(go.Bar(x=["normal", "pressure", "baseline (phase-adj.)"],
                         y=[row[f"{k[0]}_q"], row[f"{k[0]}_p"], row[f"{k[0]}_baseline_adj"]]))
    f.update_layout(title=f"{k[1]}: pressure vs normal", height=320)
    return f


def fig_phase(phase_df, player):
    import plotly.express as px
    d = phase_df[phase_df.player == player].groupby(["phase", "is_pressure"]).agg(n=("n", "sum"), voe=("voe", "sum")).reset_index()
    d["voe_per100"] = 100 * d.voe / d.n.replace(0, np.nan)
    d["condition"] = np.where(d.is_pressure, "pressure", "normal")
    return px.bar(d, x="phase", y="voe_per100", color="condition", barmode="group",
                  title="Value over expectation per 100 balls, by phase")


def fig_consistency(pm, player):
    import plotly.express as px
    d = pm[(pm.player == player) & pm.is_pressure & (pm.n > 0)].copy()
    d["voe_per100"] = 100 * d.voe / d.n
    d = d.sort_values(["season", "match_id"]).reset_index(drop=True)
    return px.bar(d, x=d.index, y="voe_per100", color="format", title="Match-by-match value over expectation (pressure balls)")


def fig_sensitivity(sens):
    import plotly.express as px
    return px.bar(sens, x="variant", y="spearman", color="side", barmode="group", title="Rank correlation vs baseline definition")


# ---------------------------------------------------------------- Matplotlib (PDF)
def _png(fig):
    import matplotlib.pyplot as plt
    b = io.BytesIO()
    fig.savefig(b, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    b.seek(0)
    return b


def _mpl_dist(states, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .pressure import pressure_score
    d = states[states.state_valid & states.W_bat.notna()]
    s = pressure_score(d, cfg.weights, "bat")
    fig, ax = plt.subplots(figsize=(6.2, 2.8))
    for c, g in d.groupby("context"):
        ax.hist(s[g.index], bins=40, alpha=.6, label=c.replace("_", " "))
    ax.axvline(s.quantile(cfg.pressure_q), color="k", ls="--", lw=1, label=f"high-pressure cut (q={cfg.pressure_q})")
    ax.set_xlabel("batting pressure score (0-100)"); ax.legend(fontsize=7)
    return _png(fig)


def _mpl_top(df, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = df.sort_values("rating", ascending=False).head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.barh(d.player, d.rating, xerr=[d.rating - d.rating_lo, d.rating_hi - d.rating], color="#3b6fb6", capsize=2)
    ax.axvline(50, color="grey", lw=.8); ax.set_title(title, fontsize=9); ax.tick_params(labelsize=7)
    return _png(fig)


def _mpl_sens(sens):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    w = .38
    for i, (s, g) in enumerate(sens.groupby("side")):
        ax.bar(np.arange(len(g)) + i * w, g.spearman, w, label={"bat": "batters", "bowl": "bowlers"}[s])
    ax.set_xticks(np.arange(len(g)) + w / 2); ax.set_xticklabels(g.variant, rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("Spearman vs baseline"); ax.legend(fontsize=7)
    return _png(fig)


# ---------------------------------------------------------------- PDF
def build_report(states, base, sens, dq, cal, cfg, manifest, path):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
    ss = getSampleStyleSheet()
    H1, H2, B = ss["Heading1"], ss["Heading2"], ss["BodyText"]
    small = ss["BodyText"].clone("small", fontSize=8, leading=10)
    S = []

    def table(df, widths=None, fs=7):
        data = [list(df.columns)] + df.astype(str).values.tolist()
        t = Table(data, colWidths=widths, repeatRows=1)
        t.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), fs), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef7")),
                               ("GRID", (0, 0), (-1, -1), .25, colors.grey), ("ALIGN", (1, 0), (-1, -1), "RIGHT")]))
        return t

    S += [Paragraph("NERV — Cricket Performance Intelligence — Methodology and Results", ss["Title"]),
          Paragraph(f"{manifest['matches']} matches, {manifest['deliveries']:,} deliveries. Config seed {cfg.seed}. "
                    "Generated automatically from the run outputs.", B), Spacer(1, 8)]
    S += [Paragraph("1. Approach", H1), Paragraph(
        "NERV — Cricket Performance Intelligence is a descriptive evaluation system: every delivery is turned into a match state, scored for pressure "
        "on a continuous 0-100 scale, and each batter/bowler is judged against the league outcome for the same format, "
        "innings phase and pressure level. Estimates are shrunk by sample size with empirical-Bayes, so a handful of "
        "extraordinary deliveries cannot produce a top ranking. No ratings are assigned by hand.", B)]
    S += [Paragraph("2. Match state and leakage control", H2), Paragraph(
        "State is reconstructed before each delivery from earlier deliveries only (score, wickets, legal balls, run rates, "
        "required rate, target). Wides and no-balls do not consume a legal ball; byes and leg-byes are team extras, not bowler "
        "runs; run-outs and retirements are not credited to the bowler; super overs are excluded; revised targets and "
        "overs are honoured when present.", B)]
    S += [Paragraph("3. Pressure score", H2), Paragraph(
        "P = 100 x (wR*R + wW*W + wT*T + wC*C). R = run-rate pressure (chase: required rate vs format reference; first "
        "innings: shortfall vs phase norm). W = wicket pressure from a calibrated wicket-resource curve (+ collapse term in "
        "first innings). T = time pressure, (1 - balls_left/balls)^2. C = contest pressure (chase: closeness of required "
        "runs to expected remaining runs; first innings: required acceleration vs par). Expected remaining runs "
        "E[b,w] = A(b)*B(w) is fitted per format on complete first innings, so 9 wickets/60 balls and 2 wickets/12 balls "
        "are compared on resources. Bowling pressure mirrors these terms from the fielding side. "
        f"Weights: {cfg.weights}. A delivery is 'high pressure' when its score is in the top {int((1 - cfg.pressure_q) * 100)}% "
        "of that format's scores.", B), Image(_mpl_dist(states, cfg), width=15 * cm, height=6.8 * cm)]
    t = pd.DataFrame([{"format": k, "innings": v["n_innings"], "par": round(v["par_total"], 1), "ref RPO": round(v["ref_rr"], 2),
                       "wicket resource B(0..10)": " ".join(f"{x:.2f}" for x in v["B"])} for k, v in cal.tables.items()])
    S += [Paragraph("Calibrated reference values", small), table(t, [2.5 * cm, 2 * cm, 1.5 * cm, 2 * cm, 9 * cm])]
    S += [Paragraph("4. Evaluation", H2), Paragraph(
        "Each delivery gets a value over expectation (VOE) in runs. Batter: (runs - E[runs]) - (dismissed - E[dismissed]) x "
        "wicket value. Bowler: -(runs charged - E[runs]) + (credited wicket - E[wicket]) x wicket value. Reference rates "
        "exclude the evaluated player; sparse format/phase/pressure cells widen to phase and then format, with the full-format "
        "rate as a last resort if no peers exist. Wicket value is the "
        "calibrated drop in expected remaining runs. Descriptive columns (pressure SR/economy, boundary and dot %, "
        "dismissal rate, delta vs a phase-adjusted baseline) are exported for every player.", B)]
    S += [Paragraph("5. Reliability and rating", H2), Paragraph(
        "Three components per player: absolute pressure value (VOE per 100 pressure balls), resilience (pressure VOE minus "
        "the player's own non-pressure VOE) and consistency (share of pressure matches with positive VOE, shrunk to the "
        "league rate). Absolute value and resilience are shrunk with lambda = tau^2/(tau^2+se^2) (equivalent to n/(n+k) with k "
        "estimated from the data) using match-clustered standard errors. Rating = "
        f"{cfg.rating_center:g} + {cfg.rating_scale:g} x z-composite with weights {cfg.rating_weights}; players need "
        f">= {cfg.min_pressure_balls} pressure balls and >= {cfg.min_pressure_matches} pressure matches. Ratings are "
        "per-ball, so no player gains from volume; intervals are approximate 90% bands.", B)]
    S.append(PageBreak())
    S.append(Paragraph("6. Results", H1))
    for side, name in (("bat", "Batters"), ("bowl", "Bowlers")):
        r = base[side]["ratings"]
        r = r[(r.group == "ALL") & r.rating.notna()].sort_values("rating", ascending=False).head(15)
        S += [Paragraph(f"Top {name} (all formats)", H2), Image(_mpl_top(r, f"{name}: rating with 90% interval"), width=14 * cm, height=8.1 * cm)]
        cols = ["player", "rank", "rating", "rating_lo", "rating_hi", "p_n", "p_matches", "confidence", "contrib_abs", "contrib_res", "contrib_cons"]
        tt = r[cols].round(2).rename(columns={"p_n": "pressure balls", "p_matches": "matches"})
        S.append(table(tt, fs=6))
        S.append(PageBreak())
    S += [Paragraph("7. Ranking stability", H1), Paragraph(
        "Players are re-rated under alternative pressure weightings, thresholds and a naive 'death overs = pressure' definition. "
        "Spearman correlation and top-N overlap against the baseline are shown below.", B), Image(_mpl_sens(sens), width=14 * cm, height=6.8 * cm),
          table(sens.round(3), fs=6)]
    S += [Paragraph("8. Data quality", H1), table(dq.head(40), [2 * cm, 4.5 * cm, 8 * cm, 1.5 * cm], fs=6)]
    S += [Paragraph("9. Limitations", H1), Paragraph(
        "Batting order and role are not controlled; players who bat late face a different pressure mix. Bowler-attributable "
        "runs on no-balls combined with byes may be slightly overstated. Opposition strength is not modelled; league "
        "expectations are format-level. The pressure weights are methodological choices, tested for stability but not "
        "uniquely identified. Rating intervals treat components as independent and are therefore approximate.", B)]
    SimpleDocTemplate(str(path), pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm, topMargin=1.5 * cm, bottomMargin=1.5 * cm).build(S)
