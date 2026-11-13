"""Synthetic cricket data in the six-CSV layout, with a *planted* pressure-resilience trait per player.

Used for (a) `--demo` runs and (b) validation tests: if NERV works, its resilience estimates
should correlate with the planted trait. Synthetic output is NOT real cricket data.
"""
import math
import random
from pathlib import Path
import pandas as pd

PHASES = {
    "pp": {0: .30, 1: .36, 2: .06, 3: .008, 4: .145, 6: .065, "W": .04},
    "mid": {0: .33, 1: .42, 2: .07, 3: .007, 4: .09, 6: .05, "W": .04},
    "death": {0: .27, 1: .34, 2: .08, 3: .006, 4: .13, 6: .12, "W": .07},
}
WK_TYPES = [("bowled", .25), ("caught", .50), ("lbw", .12), ("run out", .08), ("stumped", .03), ("caught and bowled", .02)]
COMPS = [(1, "Indian Premier League", "IPL"), (2, "T20 International", "T20I")]


def _pick(rng, probs):
    r, c = rng.random(), 0.0
    for k, p in probs:
        c += p
        if r <= c:
            return k
    return probs[-1][0]


def _players(rng, n=112):
    return {f"Player {i:03d}": {"bat": rng.gauss(0, .2), "bowl": rng.gauss(0, .2),
                                "res_bat": rng.gauss(0, .35), "res_bowl": rng.gauss(0, .35)} for i in range(n)}


def _outcome(rng, phase, bat, bowl, x, xb):
    pr = dict(PHASES[phase])
    sk = math.exp(bat["bat"] - bowl["bowl"] + bat["res_bat"] * x - bowl["res_bowl"] * xb)
    wk = math.exp(-0.7 * (bat["bat"] - bowl["bowl"]) - 0.8 * bat["res_bat"] * x + 0.8 * bowl["res_bowl"] * xb)
    for k in (4, 6):
        pr[k] *= sk
    pr[1] *= sk ** .3
    pr["W"] *= wk
    tot = sum(pr.values())
    return _pick(rng, [(k, v / tot) for k, v in pr.items()])


def _innings(rng, mid, inn, bat_team, bowl_team, squads, P, target, rows):
    xi = sorted(squads[bat_team][:11], key=lambda p: -(P[p]["bat"] + rng.gauss(0, .1)))   # better batters bat higher
    bowlers = sorted(squads[bowl_team][:11], key=lambda p: -P[p]["bowl"])[:5]
    striker, non, nxt = xi[0], xi[1], 2
    runs = wk = legal = over = 0
    while over < 20 and wk < 10:
        bowler, balls_in_over, k = bowlers[over % 5], 0, 0
        while balls_in_over < 6 and wk < 10:
            left = 120 - legal
            phase = "pp" if legal < 36 else ("mid" if legal < 96 else "death")
            if target:
                x = min(1.0, max(0.0, ((target - runs) * 6 / left - 8) / 6 + wk / 10 * .5))
            else:
                x = min(1.0, wk / 10 * 1.5 + (.2 if left <= 30 else 0))
            k += 1
            xb = (1 - x) * (1 - left / 120)          # pressure on the bowling side: batting side comfortable, innings late
            r = rng.random()
            et, ex, bruns, wtype, pout, legal_ball = "", 0, 0, "", "", True
            if r < .025:
                et, ex, legal_ball = "wides", 1, False
            elif r < .033:
                et, ex, legal_ball = "noballs", 1, False
                o = _outcome(rng, phase, P[striker], P[bowler], x, xb)
                bruns = 0 if o == "W" else o
            else:
                o = _outcome(rng, phase, P[striker], P[bowler], x, xb)
                if r < .041:
                    et, ex = "legbyes", 1
                    o = 0 if o == "W" else o
                if o == "W":
                    wtype = _pick(rng, WK_TYPES)
                    pout = non if (wtype == "run out" and rng.random() < .3) else striker
                else:
                    bruns = 0 if et else o
            total = bruns + ex
            rows.append((mid, inn, over, k, striker, non, bowler, bruns, ex, total, et, wtype, pout, bat_team, bowl_team))
            runs += total
            if legal_ball:
                legal += 1
                balls_in_over += 1
            running = bruns + (ex if et == "legbyes" else 0)
            if et == "wides":
                running = 0
            if wtype:
                wk += 1
                out_is_striker = pout == striker
                newb = xi[nxt] if nxt < 11 else None
                nxt += 1
                if newb is None or wk >= 10:
                    break
                if out_is_striker:
                    striker = newb
                else:
                    non = newb
            if running % 2 == 1:
                striker, non = non, striker
            if target and runs >= target:
                return runs
        over += 1
        striker, non = non, striker
    return runs


def generate(out_dir, n_per_comp=150, seed=7):
    rng = random.Random(seed)
    P = _players(rng)
    names = list(P)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    comps, teams, matches, rows, squads_rows, map_rows = [], [], [], [], [], []
    mid = 1000
    for cid, cname, fmt in COMPS:
        comps.append({"competition_id": cid, "name": cname, "format": fmt})
        squads = {}
        pool = names[:]
        rng.shuffle(pool)
        for t in range(8):
            tn = f"{fmt}-Team{t + 1}"
            squads[tn] = pool[t * 14:(t + 1) * 14] if fmt == "IPL" else rng.sample(names, 14)
            teams.append({"team_id": f"{cid}-{t + 1}", "team_name": tn, "competition_id": cid})
            for p in squads[tn]:
                map_rows.append({"competition_id": cid, "team_name": tn, "player_name": p})
        for i in range(n_per_comp):
            mid += 1
            a, b = rng.sample(list(squads), 2)
            season = rng.choice([2021, 2022, 2023])
            for sq in (a, b):
                rng.shuffle(squads[sq])
            r1 = _innings(rng, mid, 1, a, b, squads, P, None, rows)
            r2 = _innings(rng, mid, 2, b, a, squads, P, r1 + 1, rows)
            matches.append({"match_id": mid, "competition_id": cid, "season": season, "date": f"{season}-04-{1 + i % 28:02d}",
                            "team1": a, "team2": b, "winner": b if r2 > r1 else a})
            for sq in (a, b):
                for p in squads[sq][:11]:
                    squads_rows.append({"match_id": mid, "team_name": sq, "player_name": p})
    cols = ["match_id", "innings", "over", "ball", "batter", "non_striker", "bowler", "runs_batter", "runs_extras",
            "runs_total", "extras_type", "wicket_type", "player_out", "batting_team", "bowling_team"]
    pd.DataFrame(rows, columns=cols).to_csv(out / "ball_by_ball.csv", index=False)
    pd.DataFrame(comps).to_csv(out / "competitions.csv", index=False)
    pd.DataFrame(teams).to_csv(out / "teams.csv", index=False)
    pd.DataFrame(matches).to_csv(out / "matches.csv", index=False)
    pd.DataFrame(squads_rows).to_csv(out / "playing_eleven.csv", index=False)
    pd.DataFrame(map_rows).to_csv(out / "competition_team_player_mapping.csv", index=False)
    truth = pd.DataFrame([{"player": k, **v} for k, v in P.items()])
    return truth
