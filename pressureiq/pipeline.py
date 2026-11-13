"""End-to-end orchestration:  python -m pressureiq.pipeline [--raw data/raw] [--demo]"""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from .config import Config, ALT_WEIGHTS, ROOT
from .ingest import load_and_clean
from .state import build_states
from .pressure import fit_calibration, compute_components
from .evaluate import run_side


def sensitivity(states, cfg, base, topn=(10, 20)):
    """Re-rate every player under alternative pressure definitions / thresholds and measure rank stability."""
    variants = [(f"weights:{k}", dict(weights=w)) for k, w in ALT_WEIGHTS.items()]
    variants += [(f"threshold:q={q}", dict(q=q)) for q in (0.60, 0.70, 0.80, 0.90)]
    variants += [("naive:death_overs", dict(naive=True))]
    out = []
    for side in ("bat", "bowl"):
        b = base[side]["ratings"]
        b = b[(b.group == "ALL") & b.rating.notna()].set_index("player")
        for name, kw in variants:
            r = run_side(states, side, cfg, **kw)["ratings"]
            r = r[(r.group == "ALL") & r.rating.notna()].set_index("player")
            common = b.index.intersection(r.index)
            row = {"side": side, "variant": name, "n_common": len(common), "n_eligible": len(r)}
            if len(common) > 3:
                row["spearman"] = float(spearmanr(b.loc[common, "rating"], r.loc[common, "rating"])[0])
                row["mean_abs_rank_shift"] = float((b.loc[common, "rating"].rank() - r.loc[common, "rating"].rank()).abs().mean())
            for n in topn:
                tb, tr = set(b.rating.nlargest(n).index), set(r.rating.nlargest(n).index)
                row[f"top{n}_overlap"] = len(tb & tr) / n
            out.append(row)
    return pd.DataFrame(out)


def _hash_inputs(raw_dir):
    h = {}
    for p in sorted(Path(raw_dir).glob("*.csv")):
        h[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    return h


def run_pipeline(raw_dir, proc_dir, out_dir, cfg=None, make_pdf=True, log=print):
    cfg = cfg or Config()
    t0 = time.time()
    proc_dir, out_dir = Path(proc_dir), Path(out_dir)
    proc_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    log("01 ingest ...")
    balls, matches, dq = load_and_clean(raw_dir)
    log("02 match-state reconstruction ...")
    states = build_states(balls, matches, cfg)
    n_inc = states[~states.innings_complete].groupby(["match_id", "innings"]).ngroups
    if n_inc:
        dq.add("warning", "incomplete innings", "kept for evaluation, excluded from pressure calibration", n_inc)
    dq.add("info", "deliveries excluded (invalid state)", "chase without target or beyond ball limit", int((~states.state_valid).sum()))
    log("03 pressure calibration + scoring ...")
    cal = fit_calibration(states, cfg)
    states = states.join(compute_components(states, cal, cfg))
    states.to_parquet(proc_dir / "states.parquet", index=False)
    (proc_dir / "calibration.json").write_text(json.dumps(cal.to_json(), indent=2))
    log("04-06 batting / bowling evaluation + reliability ...")
    base = {s: run_side(states, s, cfg) for s in ("bat", "bowl")}
    log("07 sensitivity analysis ...")
    sens = sensitivity(states, cfg, base)
    for s, name in (("bat", "batting_ratings"), ("bowl", "bowling_ratings")):
        r = base[s]["ratings"]
        r.to_parquet(out_dir / f"{name}.parquet", index=False)
        r.sort_values(["group", "rank"]).to_csv(out_dir / f"{name}.csv", index=False)
        base[s]["player_match"].to_parquet(out_dir / f"{s}_player_match.parquet", index=False)
        base[s]["phase"].to_parquet(out_dir / f"{s}_phase.parquet", index=False)
    sens.to_csv(out_dir / "sensitivity.csv", index=False)
    dqdf = dq.to_df()
    dqdf.to_csv(out_dir / "data_quality.csv", index=False)
    manifest = {"config": cfg.to_dict(), "inputs_sha256": _hash_inputs(raw_dir), "matches": int(states.match_id.nunique()),
                "deliveries": int(len(states)), "python": platform.python_version(), "pandas": pd.__version__,
                "numpy": np.__version__, "runtime_sec": round(time.time() - t0, 1)}
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    if make_pdf:
        from .report import build_report
        log("08 methodology report ...")
        build_report(states, base, sens, dqdf, cal, cfg, manifest, out_dir / "NERV_methodology.pdf")
    log(f"done in {time.time() - t0:.1f}s -> {out_dir}")
    return states, base, sens, dqdf, cal


def main():
    ap = argparse.ArgumentParser(description="NERV — Cricket Performance Intelligence pipeline")
    ap.add_argument("--raw", default=str(ROOT / "data/raw"))
    ap.add_argument("--processed", default=str(ROOT / "data/processed"))
    ap.add_argument("--out", default=str(ROOT / "data/outputs"))
    ap.add_argument("--demo", action="store_true", help="generate SYNTHETIC data under data/demo and run on it")
    ap.add_argument("--demo-matches", type=int, default=150, help="matches per competition for --demo")
    ap.add_argument("--no-pdf", action="store_true")
    a = ap.parse_args()
    if a.demo:
        from .synth import generate
        d = ROOT / "data/demo"
        generate(d / "raw", n_per_comp=a.demo_matches)
        a.raw, a.processed, a.out = str(d / "raw"), str(d / "processed"), str(d / "outputs")
        print("NOTE: demo mode uses synthetic data; the resulting rankings are not real.")
    run_pipeline(a.raw, a.processed, a.out, make_pdf=not a.no_pdf)


if __name__ == "__main__":
    main()
