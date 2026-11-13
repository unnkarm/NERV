"""NERV — Cricket Performance Intelligence dashboard:  streamlit run dashboard.py
Reads the pipeline's parquet outputs through DuckDB. Filters re-rate players on the filtered evidence
using the same engine as the pipeline (league expectations and pressure thresholds stay fixed)."""
import os
from pathlib import Path
import duckdb
import pandas as pd
import streamlit as st
from pressureiq.config import Config, ROOT
from pressureiq.evaluate import prepare_rows, aggregate_side
from pressureiq import report as rp

st.set_page_config(page_title="NERV — Cricket Performance Intelligence", layout="wide")
cfg = Config()

base = Path(os.environ.get("PRESSUREIQ_DATA", ROOT / "data"))
proc, out = base / "processed", base / "outputs"
if not (proc / "states.parquet").exists() and (ROOT / "data/demo/processed/states.parquet").exists():
    base = ROOT / "data/demo"
    proc, out = base / "processed", base / "outputs"
    st.warning("Showing SYNTHETIC demo data (data/demo). Run the pipeline on data/raw for real results.")
if not (proc / "states.parquet").exists():
    st.error("No pipeline outputs found. Run:  python -m pressureiq.pipeline")
    st.stop()


@st.cache_data(show_spinner="Loading states ...")
def load_states():
    con = duckdb.connect()
    return con.execute(f"select * from read_parquet('{(proc / 'states.parquet').as_posix()}')").df()


@st.cache_data(show_spinner="Scoring deliveries ...")
def prepared(side, q):
    return prepare_rows(load_states(), side, cfg, q=q)


states = load_states()
st.title("NERV — Cricket Performance Intelligence")
c = st.columns(4)
c[0].metric("Matches analysed", f"{states.match_id.nunique():,}")
c[1].metric("Deliveries", f"{len(states):,}")
c[2].metric("Formats", ", ".join(sorted(states.format.unique())))
c[3].metric("Seasons", f"{states.season.min()}-{states.season.max()}")

with st.sidebar:
    side = st.radio("Role", ["bat", "bowl"], format_func=lambda s: "Batters" if s == "bat" else "Bowlers")
    q = st.slider("Pressure threshold (top share of deliveries)", 0.50, 0.95, cfg.pressure_q, 0.05,
                  help="0.75 = the top 25% of a format's pressure scores count as high pressure")
    fmts = st.multiselect("Format", sorted(states.format.unique()))
    seasons = st.multiselect("Season", sorted(states.season.unique()))
    tcol = "batting_team" if side == "bat" else "bowling_team"
    teams = st.multiselect("Team", sorted(t for t in states[tcol].dropna().unique() if t)) if tcol in states else []
    min_balls = st.number_input("Min pressure balls", 10, 1000, cfg.min_pressure_balls, 10)
cfg.min_pressure_balls = int(min_balls)

rows = prepared(side, q)
res = aggregate_side(rows, side, cfg, {"format": fmts, "season": seasons, "team": teams})
R = res["ratings"]
if R.empty:
    st.error("No player data is available for the selected dataset and filters.")
    st.stop()
grp = st.sidebar.selectbox("Rating population", sorted(R.group.unique(), key=lambda g: (g != "ALL", g)))
tab1, tab2, tab3 = st.tabs(["Rankings", "Player deep dive", "Model diagnostics"])

with tab1:
    r = R[(R.group == grp) & R.rating.notna()].sort_values("rating", ascending=False)
    st.caption(f"{len(r)} eligible players. 50 = average; bars show approximate 90% intervals.")
    st.plotly_chart(rp.fig_rank_bars(r, "Pressure rating"), width="stretch")
    show = ["rank", "player", "rating", "rating_lo", "rating_hi", "confidence_tier", "p_n", "p_matches",
            "contrib_abs", "contrib_res", "contrib_cons"]
    show += ["sr_p", "delta_sr_adj", "wkt_per100_p"] if side == "bat" else ["econ_p", "delta_econ_adj", "wkt_per100_p"]
    st.dataframe(r[show].round(2), width="stretch", hide_index=True)
    st.download_button("Download CSV", r.to_csv(index=False), f"nerv_{side}_{grp}.csv")

with tab2:
    players = sorted(R.player.unique())
    p = st.selectbox("Player", players)
    pr = R[(R.player == p)]
    st.dataframe(pr.dropna(axis=1, how="all").round(2).T.astype(str) if len(pr) else pr, width="stretch")
    row = pr[pr.group == grp]
    if len(row) and row.iloc[0].n_p > 0 and row.iloc[0].n_q > 0:
        a, b = st.columns(2)
        a.plotly_chart(rp.fig_pressure_vs_normal(row.iloc[0], side), width="stretch")
        b.plotly_chart(rp.fig_phase(res["phase"], p), width="stretch")
    st.plotly_chart(rp.fig_consistency(res["player_match"], p), width="stretch")
    if len(pr[pr.group != "ALL"]) > 1:
        st.subheader("Format comparison")
        st.dataframe(pr[pr.group != "ALL"][["group", "rating", "p_n", "abs_adj", "res_adj", "confidence_tier"]].round(2), hide_index=True)

with tab3:
    a, b = st.columns(2)
    a.plotly_chart(rp.fig_pressure_distribution(states, side), width="stretch")
    b.plotly_chart(rp.fig_components(states), width="stretch")
    sp = out / "sensitivity.csv"
    if sp.exists():
        st.plotly_chart(rp.fig_sensitivity(pd.read_csv(sp)), width="stretch")
        st.dataframe(pd.read_csv(sp).round(3), hide_index=True)
    dqp = out / "data_quality.csv"
    if dqp.exists():
        st.subheader("Data-quality warnings")
        st.dataframe(pd.read_csv(dqp), hide_index=True, width="stretch")
