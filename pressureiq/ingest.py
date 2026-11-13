"""Stage 01: load the six CSVs, map schemas, clean, validate, and produce a data-quality report."""
from pathlib import Path
import re
import numpy as np
import pandas as pd
from .config import BBB_ALIASES, MATCH_ALIASES, COMP_ALIASES, PLAYER_ALIASES, BOWLER_CREDITED

FILES = ["competitions", "teams", "matches", "ball_by_ball", "playing_eleven", "competition_team_player_mapping"]
REQUIRED_BBB = ["match_id", "innings", "over", "batter", "bowler", "runs_batter"]
WICKET_ALIASES = {
    "runout": "run out",
    "stump": "stumped",
    "retire out": "retired out",
    "obstracting the field": "obstructing the field",
}


class SchemaError(ValueError):
    pass


class DataQuality:
    """Collects audit findings (Phase 1 deliverable)."""

    def __init__(self):
        self.items = []

    def add(self, severity, check, detail, count=0):
        self.items.append({"severity": severity, "check": check, "detail": detail, "count": int(count)})

    def to_df(self):
        return pd.DataFrame(self.items, columns=["severity", "check", "detail", "count"])


def _rename(df, aliases):
    low = {re.sub(r"\s+", "_", c.strip().lower()): c for c in df.columns}
    ren = {}
    for canon, alts in aliases.items():
        for a in [canon, *alts]:
            if a in low and low[a] not in ren:
                ren[low[a]] = canon
                break
    return df.rename(columns=ren)


def load_raw(raw_dir):
    out, missing = {}, []
    for name in FILES:
        p = Path(raw_dir) / f"{name}.csv"
        if p.exists():
            out[name] = pd.read_csv(p, low_memory=False)
        else:
            missing.append(name)
    return out, missing


def _norm_extras(s):
    s = str(s).lower()
    if s in ("", "nan", "none"):
        return ""
    tags = []
    if "wide" in s:
        tags.append("wides")
    if re.search(r"no[\s_-]?ball", s) or "overstep" in s:
        tags.append("noballs")
    if re.search(r"leg[\s_-]?bye", s):
        tags.append("legbyes")
    elif "bye" in s:
        tags.append("byes")
    if "penalty" in s:
        tags.append("penalty")
    return ",".join(tags) if tags else "other"


def _norm_wicket(s):
    s = str(s).lower().replace("_", " ").strip()
    return "" if s in ("", "nan", "none", "0") else WICKET_ALIASES.get(s, s)


def infer_format(text):
    t = str(text).lower()
    if "ipl" in t or "indian premier" in t:
        return "IPL"
    if "t20i" in t or "international" in t:
        return "T20I"
    return str(text).strip().upper() if str(text).strip() not in ("", "nan") else "T20"


def clean_ball_by_ball(raw, dq):
    raw = raw.copy()
    low = {re.sub(r"\s+", "_", c.strip().lower()): c for c in raw.columns}
    extras_parts = [low[c] for c in ("extras1_runs", "extras_2_runs") if c in low]
    extras_aliases = {"runs_extras", *BBB_ALIASES["runs_extras"]}
    if extras_parts and not extras_aliases.intersection(low):
        raw["runs_extras"] = sum(pd.to_numeric(raw[c], errors="coerce").fillna(0) for c in extras_parts)
    batter_run_aliases = {"runs_batter", *BBB_ALIASES["runs_batter"]}
    if "runs" in low and extras_parts and not batter_run_aliases.intersection(low):
        raw = raw.rename(columns={low["runs"]: "runs_batter"})
    df = _rename(raw, BBB_ALIASES)
    missing = [c for c in REQUIRED_BBB if c not in df.columns]
    if missing:
        raise SchemaError(f"ball_by_ball.csv is missing required columns {missing}. "
                          f"Found: {list(raw.columns)}. Add aliases in pressureiq/config.py.")
    df = df.copy()
    df["row_order"] = np.arange(len(df))               # file order defines delivery order
    df["match_id"] = df["match_id"].astype(str).str.strip()
    for c in ["innings", "over", "ball", "runs_batter", "runs_extras", "runs_total"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    bad = df[["innings", "runs_batter"]].isna().any(axis=1)
    if bad.any():
        dq.add("warning", "non-numeric key fields", "rows dropped: innings or runs_batter not numeric", bad.sum())
        df = df[~bad]
    df["runs_batter"] = df["runs_batter"].fillna(0)
    if "runs_extras" not in df.columns:
        df["runs_extras"] = (df["runs_total"] - df["runs_batter"]) if "runs_total" in df.columns else 0.0
        dq.add("info", "runs_extras derived", "runs_extras missing; derived from runs_total - runs_batter")
    df["runs_extras"] = df["runs_extras"].fillna(0)
    if "runs_total" not in df.columns:
        df["runs_total"] = df["runs_batter"] + df["runs_extras"]
    df["runs_total"] = df["runs_total"].fillna(df["runs_batter"] + df["runs_extras"])
    if "extras_type" not in df.columns:
        df["extras_type"] = ""
        dq.add("warning", "extras_type missing", "wides/no-balls cannot be identified: every delivery treated as legal")
    df["extras_type"] = df["extras_type"].map(_norm_extras)
    if "wicket_type" not in df.columns:
        df["wicket_type"] = ""
        dq.add("warning", "wicket_type missing", "no dismissals can be identified")
    df["wicket_type"] = df["wicket_type"].map(_norm_wicket)
    for c in ["batter", "bowler", "non_striker", "player_out", "batting_team", "bowling_team"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().replace({"nan": "", "None": ""})
    if "player_out" not in df.columns:
        df["player_out"] = ""
    return df


def clean_matches(raw, comps, balls, dq):
    if raw is None:
        dq.add("warning", "matches.csv missing", "match table rebuilt from ball_by_ball; format defaults to T20")
        m = pd.DataFrame({"match_id": balls["match_id"].unique()})
    else:
        m = _rename(raw, MATCH_ALIASES)
        if "match_id" not in m.columns:
            raise SchemaError("matches.csv has no match_id column")
    m["match_id"] = m["match_id"].astype(str).str.strip()
    if m["match_id"].duplicated().any():
        dq.add("warning", "duplicate match_id", "duplicates dropped (first kept)", m["match_id"].duplicated().sum())
        m = m.drop_duplicates("match_id")
    fmt = None
    if "format" in m.columns:
        fmt = m["format"]
    elif comps is not None and "competition_id" in m.columns:
        c = _rename(comps, COMP_ALIASES)
        c["competition_id"] = c["competition_id"].astype(str)
        m["competition_id"] = m["competition_id"].astype(str)
        src = "format" if "format" in c.columns else ("name" if "name" in c.columns else None)
        if src:
            m = m.merge(c[["competition_id", src]].rename(columns={src: "_f"}), on="competition_id", how="left")
            fmt = m["_f"]
    m["format"] = fmt.map(infer_format) if fmt is not None else "T20"
    if fmt is None:
        dq.add("warning", "format unknown", "no format / competition information: all matches labelled T20")
    if "season" not in m.columns:
        m["season"] = pd.to_datetime(m["date"], errors="coerce").dt.year.astype("Int64").astype(str) if "date" in m.columns else "NA"
    m["season"] = m["season"].astype(str)
    for c in ["target_runs", "target_overs", "overs"]:
        if c in m.columns:
            m[c] = pd.to_numeric(m[c], errors="coerce")
    return m.drop(columns=[c for c in ["_f"] if c in m.columns])


def map_players(balls, playing_eleven, dq):
    """Entity mapping: if ball_by_ball stores numeric player ids and playing_eleven can resolve them, use names."""
    if playing_eleven is None:
        return balls
    pe = _rename(playing_eleven, PLAYER_ALIASES)
    if {"player_id", "player_name"} <= set(pe.columns):
        def player_key(value):
            s = str(value).strip()
            try:
                n = float(s)
                if n.is_integer():
                    return str(int(n))
            except ValueError:
                pass
            return s

        mp = {player_key(pid): str(name).strip() for pid, name in zip(pe.player_id, pe.player_name)
              if pd.notna(pid) and pd.notna(name) and str(name).strip()}
        mapped_out = 0
        if "player_out" in balls:
            original = balls["player_out"]
            balls["player_out"] = original.map(lambda value: mp.get(player_key(value), value))
            mapped_out = int((balls["player_out"] != original).sum())
        if balls["batter"].map(player_key).isin(mp.keys()).mean() > 0.9:
            for c in ["batter", "bowler", "non_striker", "player_out"]:
                if c in balls.columns:
                    balls[c] = balls[c].map(lambda value: mp.get(player_key(value), value))
            dq.add("info", "player ids mapped", "player ids resolved to names")
        elif mapped_out:
            dq.add("info", "dismissed player ids mapped", "dismissed player ids resolved to names")
    return balls


def validate(balls, matches, dq):
    n = len(balls)
    dq.add("info", "rows", f"{n} deliveries, {balls.match_id.nunique()} matches", n)
    d = balls.duplicated(subset=[c for c in balls.columns if c != "row_order"]).sum()
    if d:
        dq.add("warning", "exact duplicate deliveries", "identical rows (possible double load)", d)
    for c in ["batter", "bowler"]:
        e = (balls[c] == "").sum()
        if e:
            dq.add("error", f"empty {c}", "deliveries with no player name", e)
    unk = set(balls.match_id) - set(matches.match_id)
    if unk:
        dq.add("warning", "matches missing from matches.csv", "assigned format T20", len(unk))
    s = int((balls.innings > 2).sum()) + (int(balls["is_super_over"].astype(bool).sum()) if "is_super_over" in balls else 0)
    if s:
        dq.add("info", "super-over deliveries", "excluded from evaluation", s)
    neg = (balls[["runs_batter", "runs_extras", "runs_total"]] < 0).any(axis=1).sum()
    if neg:
        dq.add("error", "negative runs", "check source", neg)
    mism = (balls.runs_total - balls.runs_batter - balls.runs_extras).abs() > 0
    if mism.sum():
        dq.add("warning", "runs_total != batter + extras", "runs_total used for team score", mism.sum())
    wk = balls.wicket_type != ""
    no_out = (wk & (balls.player_out == "")).sum()
    if no_out:
        dq.add("warning", "dismissal without player_out", "assumed striker dismissed", no_out)
    odd_ex = (balls.extras_type == "other").sum()
    if odd_ex:
        dq.add("warning", "unrecognised extras_type", "treated as legal deliveries", odd_ex)
    unknown_w = (~balls.wicket_type[wk].isin(BOWLER_CREDITED | {"run out", "retired out", "obstructing the field",
                                                                  "retired hurt", "retired not out", "timed out",
                                                                  "handled the ball", "hit the ball twice"})).sum()
    if unknown_w:
        dq.add("warning", "unrecognised wicket_type", "counted as team wicket, not credited to bowler", unknown_w)
    per = balls[balls.innings <= 2].groupby("match_id").innings.nunique()
    if (per < 2).sum():
        dq.add("warning", "matches with a single innings", "chase state unavailable; chase rows excluded", (per < 2).sum())
    return dq


def load_and_clean(raw_dir):
    dq = DataQuality()
    tables, missing = load_raw(raw_dir)
    if "ball_by_ball" not in tables:
        raise FileNotFoundError(f"{Path(raw_dir) / 'ball_by_ball.csv'} not found")
    for m in missing:
        dq.add("info", f"{m}.csv not found", "optional for the evaluation engine" if m != "matches" else "match table rebuilt")
    balls = clean_ball_by_ball(tables["ball_by_ball"], dq)
    player_lookup = tables.get("playing_eleven")
    if player_lookup is None or not {"player_id", "player_name"} <= set(_rename(player_lookup, PLAYER_ALIASES).columns):
        player_lookup = tables.get("competition_team_player_mapping", player_lookup)
    balls = map_players(balls, player_lookup, dq)
    matches = clean_matches(tables.get("matches"), tables.get("competitions"), balls, dq)
    validate(balls, matches, dq)
    return balls, matches, dq
