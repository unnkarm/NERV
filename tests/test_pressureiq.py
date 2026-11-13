import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr
from pressureiq.config import Config
from pressureiq.ingest import DataQuality, clean_ball_by_ball, clean_matches, map_players, SchemaError
from pressureiq.state import build_states
from pressureiq.pressure import fit_calibration, compute_components
from pressureiq.evaluate import aggregate_side, prepare_rows, run_side
from pressureiq.reliability import eb_shrink
from pressureiq.synth import generate

CFG = Config()


def _balls(rows):
    cols = ["match_id", "innings", "over", "ball", "batter", "bowler", "runs_batter", "runs_extras", "runs_total",
            "extras_type", "wicket_type", "player_out"]
    return pd.DataFrame(rows, columns=cols)


def _states(rows):
    dq = DataQuality()
    b = clean_ball_by_ball(_balls(rows), dq)
    m = clean_matches(None, None, b, dq)
    return build_states(b, m, CFG)


# ----------------------------------------------------------------------------- data / state
def test_missing_required_columns_raise():
    with pytest.raises(SchemaError):
        clean_ball_by_ball(pd.DataFrame({"match_id": [1]}), DataQuality())


def test_maps_dataset_ball_by_ball_schema():
    raw = pd.DataFrame({
        "match_id": ["m", "m", "m", "m"], "innings": [1, 1, 1, 1], "overid": [1, 1, 1, 1],
        "striker_name": ["A", "A", "A", "A"], "bowler_name": ["X", "X", "X", "X"],
        "runs": [4, 0, 0, 0], "extras_type": [None, "wide", "Overstep", None],
        "extras1_runs": [0, 1, 1, 0], "extras_2_runs": [0, 0, 4, 0],
        "dismissal_type": [None, None, None, "RunOut"], "dismissed_player_id": [None, None, None, 7],
    })
    balls = clean_ball_by_ball(raw, DataQuality())
    assert balls.over.tolist() == [1, 1, 1, 1]
    assert balls.runs_batter.tolist() == [4, 0, 0, 0]
    assert balls.runs_extras.tolist() == [0, 1, 5, 0]
    assert balls.runs_total.tolist() == [4, 1, 5, 0]
    assert balls.extras_type.tolist() == ["", "wides", "noballs", ""]
    balls = map_players(balls, pd.DataFrame({"player_id": [7], "player_name": ["A"]}), DataQuality())
    assert balls.loc[3, "player_out"] == "A"
    assert balls.loc[3, "wicket_type"] == "run out"


def test_maps_dataset_match_schema():
    raw = pd.DataFrame({
        "match_id": [1], "comp_id": [9], "match_date": ["2026-01-01"],
        "revised_target": [151], "first_inn_revised_overs": [18],
        "second_inn_revised_overs": [17],
    })
    comps = pd.DataFrame({"comp_id": [9], "match_format": ["T20I"]})
    balls = pd.DataFrame({"match_id": ["1"]})
    matches = clean_matches(raw, comps, balls, DataQuality())
    assert matches.loc[0, "format"] == "T20I"
    assert matches.loc[0, "target_runs"] == 151
    assert matches.loc[0, "overs"] == 18
    assert matches.loc[0, "target_overs"] == 17


def test_legal_balls_extras_and_wickets():
    rows = [
        ("m", 1, 0, 1, "A", "X", 0, 1, 1, "wides", "", ""),            # wide: no legal ball
        ("m", 1, 0, 2, "A", "X", 4, 0, 4, "", "", ""),
        ("m", 1, 0, 3, "A", "X", 0, 1, 1, "legbyes", "", ""),          # leg-bye: legal, team run
        ("m", 1, 0, 4, "A", "X", 0, 1, 1, "noballs", "", ""),          # no-ball: not legal
        ("m", 1, 0, 5, "A", "X", 0, 0, 0, "", "run out", "A"),         # run out: team wicket, not bowler's
        ("m", 1, 0, 6, "B", "X", 0, 0, 0, "", "retired hurt", "B"),    # not a wicket
        ("m", 1, 0, 7, "B", "X", 0, 0, 0, "", "bowled", "B"),
    ]
    s = _states(rows)
    assert list(s.legal) == [False, True, True, False, True, True, True]
    assert list(s.legal_before) == [0, 0, 1, 2, 2, 3, 4]
    assert list(s.runs_before) == [0, 1, 5, 6, 7, 7, 7]
    assert list(s.wickets_before) == [0, 0, 0, 0, 0, 1, 1]
    assert list(s.bowler_wicket) == [False] * 4 + [False, False, True]
    assert s.bowler_runs.tolist() == [1, 4, 0, 1, 0, 0, 0]            # leg-bye is a team extra


def test_target_and_no_leakage():
    r1 = [("m", 1, 0, i + 1, "A", "X", 1, 0, 1, "", "", "") for i in range(6)]
    r2 = [("m", 2, 0, i + 1, "C", "Y", 2, 0, 2, "", "", "") for i in range(6)]
    s = _states(r1 + r2)
    c = s[s.innings == 2]
    assert (c.target == 7).all() and c.runs_required.iloc[0] == 7
    # changing the FUTURE must not change an earlier delivery's state
    r2b = r2[:5] + [("m", 2, 0, 6, "C", "Y", 6, 0, 6, "", "bowled", "C")]
    c2 = _states(r1 + r2b)
    cols = ["runs_before", "wickets_before", "balls_remaining", "rrr", "runs_required"]
    pd.testing.assert_frame_equal(c[cols].iloc[:5].reset_index(drop=True), c2[c2.innings == 2][cols].iloc[:5].reset_index(drop=True))


def test_super_overs_excluded():
    r = [("m", 1, 0, 1, "A", "X", 1, 0, 1, "", "", ""), ("m", 3, 0, 1, "A", "X", 6, 0, 6, "", "", "")]
    assert set(_states(r).innings) == {1}


# ----------------------------------------------------------------------------- reliability
def test_shrinkage_pulls_small_samples_harder():
    # Player A: huge estimate, tiny sample; Player B: modest estimate, large sample (spec example)
    obs = np.r_[np.random.default_rng(0).normal(0, 3, 60), 25.0, 8.0]
    se = np.r_[np.full(60, 4.0), 20.0, 2.0]
    post, sd, lam, mu, tau2 = eb_shrink(obs, se)
    assert lam[-2] < lam[-1]
    assert post[-2] < obs[-2] * 0.5 and abs(post[-1] - obs[-1]) < abs(post[-2] - obs[-2])


# ----------------------------------------------------------------------------- end-to-end on synthetic data
@pytest.fixture(scope="module")
def world(tmp_path_factory):
    d = tmp_path_factory.mktemp("raw")
    truth = generate(d, n_per_comp=90, seed=3)
    from pressureiq.ingest import load_and_clean
    balls, matches, dq = load_and_clean(d)
    states = build_states(balls, matches, CFG)
    cal = fit_calibration(states, CFG)
    states = states.join(compute_components(states, cal, CFG))
    return states, truth, cal


def test_pressure_bounds_and_monotonicity(world):
    states, _, cal = world
    v = states[states.state_valid]
    from pressureiq.pressure import pressure_score
    for side in ("bat", "bowl"):
        s = pressure_score(v, CFG.weights, side)
        assert s.between(0, 100).all()
    comps = v[["R_bat", "W_bat", "T", "C_bat", "R_bowl", "W_bowl", "C_bowl"]]
    assert comps.min().min() >= 0 and comps.max().max() <= 1 + 1e-9
    # wicket pressure rises with wickets lost; time pressure rises as balls run out
    assert v.groupby("wickets_before").W_bat.mean().is_monotonic_increasing
    assert v.groupby("balls_remaining").T.mean().is_monotonic_decreasing
    B = cal.get("IPL")["B"]
    assert np.all(np.diff(B) <= 1e-12)          # wicket resource never increases with wickets lost


def test_resources_not_raw_required_rate(world):
    """Same required rate, different resources -> different pressure (spec section 3B)."""
    states, _, _ = world
    c = states[(states.context == "chase") & states.state_valid]
    a = c[(c.wickets_before <= 1) & (c.balls_remaining.between(55, 65))].W_bat.mean()
    b = c[(c.wickets_before >= 5) & (c.balls_remaining.between(55, 65))].W_bat.mean()
    assert b > a


def test_reproducible(world):
    states, _, _ = world
    r1 = run_side(states, "bat", CFG)["ratings"].sort_values(["group", "player"]).reset_index(drop=True)
    r2 = run_side(states, "bat", CFG)["ratings"].sort_values(["group", "player"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(r1, r2)


def test_no_pressure_rows_keep_rating_schema(world):
    states, _, _ = world
    rows = prepare_rows(states, "bat", CFG)
    rows["is_pressure"] = False
    rows["level"] = "low"
    ratings = aggregate_side(rows, "bat", CFG)["ratings"]
    assert {"rating", "rating_lo", "rating_hi", "confidence_tier", "rank"}.issubset(ratings.columns)
    assert ratings.rating.isna().all()


@pytest.mark.parametrize("side,col", [("bat", "res_bat"), ("bowl", "res_bowl")])
def test_recovers_planted_pressure_trait(world, side, col):
    states, truth, _ = world
    r = run_side(states, side, CFG)["ratings"]
    r = r[(r.group == "ALL") & r.rating.notna()].merge(truth, on="player")
    assert len(r) > 25
    assert spearmanr(r.rating, r[col])[0] > 0.15


def test_rating_is_additive_and_volume_free(world):
    states, _, _ = world
    r = run_side(states, "bat", CFG)["ratings"]
    r = r[(r.group == "ALL") & r.rating.notna()]
    assert np.allclose(r.rating, CFG.rating_center + r.contrib_abs + r.contrib_res + r.contrib_cons)
    assert (r.rating_lo <= r.rating).all() and (r.rating <= r.rating_hi).all()
    assert (r.p_n >= CFG.min_pressure_balls).all()                      # eligibility enforced
    assert abs(spearmanr(r.rating, r.p_matches)[0]) < 0.5               # no strong exposure advantage
