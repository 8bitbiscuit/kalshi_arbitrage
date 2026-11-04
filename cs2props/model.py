"""Features, training and prediction for CS2 "Maps 1-2" kill / headshot props.

    python -m cs2props.model train     # data/maps.csv  -> data/model_{kills,headshots}.joblib
    python -m cs2props.model predict   # data/lines.csv -> data/picks.csv
    python -m cs2props.model backtest [data/prop_history.csv | data/lines_history.csv] [gbm | tree]
    python -m cs2props.model compare   # gradient boosting vs a single decision tree, same test data

The model (gradient-boosted trees by default) predicts the stat itself (e.g. maps 1-2 kills). There are
no historical prop lines to label "over/under" with, so P(over line) comes from the projection plus the
spread of its held-out errors.
Any line, from any book, can then be scored without retraining.
"""
import difflib
import os
import re
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor

MAPS_CSV, LINES_CSV, PICKS_CSV = "data/maps.csv", "data/lines.csv", "data/picks.csv"
PROPS_CSV, BACKTEST_CSV, CAL_JOBLIB = "data/prop_history.csv", "data/backtest.csv", "data/calibration.joblib"
MIN_CAL_LINES = 300  # graded lines from earlier months needed before a month gets a calibrated score
STATS = ("kills", "headshots")
FEATURES = ["kills_l20", "kills_l5", "headshots_l20", "hs_pct", "kpr", "adr_l20", "kast_l20", "rounds_l20",
            "maps_played", "map_factor", "win_prob", "elo_opp", "sched_gap"]
# Rows lacking these are skipped (fewer than 5 maps of history). Other features may be NaN (e.g. a
# missing ADR); scikit-learn trees handle NaN natively.
NEED = {"kills": ["kills_l20", "kpr", "map_factor"], "headshots": ["kills_l20", "kpr", "map_factor", "hs_pct"]}
PRIOR_ROUNDS = 100   # shrink a player's per-map KPR toward their recent KPR by this many rounds
POOL = 15            # a team's last 15 maps stand in for the unknown veto
DEFAULT_ODDS = -119  # lines without a price (PrizePicks): breakeven 54.3%, about a 5/6-pick flex entry


def american_to_prob(odds):
    return 100 / (odds + 100) if odds > 0 else -odds / (-odds + 100)


def probs(pred, line, z):
    """(P(over), P(under)) for an integer stat = round(projection + z * sqrt(projection)).

    z are held-out errors (actual - pred) / sqrt(pred), centred to mean 0: counting stats spread like
    sqrt(mean), so one z distribution fits both 15-kill and 35-kill projections.
    """
    z, pred, line = np.sort(z), np.maximum(np.asarray(pred, float), 1e-9), np.asarray(line, float)
    sd = np.sqrt(pred)
    over = 1 - np.searchsorted(z, (np.floor(line) + 0.5 - pred) / sd, side="right") / len(z)
    under = np.searchsorted(z, (np.ceil(line) - 0.5 - pred) / sd, side="left") / len(z)
    return over, under


def load(path=MAPS_CSV):
    df = pd.read_csv(path)
    df["start_date"] = pd.to_datetime(df.start_date, utc=True, format="ISO8601")
    df = df.dropna(subset=["player_id"]).astype({"player_id": int}).drop_duplicates(["game_id", "player_id"])
    # Keep only complete box scores. bo3.gg rows without headshots are partial live snapshots (their
    # kills run ~20% low); headshots > kills is a known glitch (2025-10-27); a full map lasts 13+ rounds
    # and has ~6.7 kills per round in total.
    bad = (df.headshots.isna() | (df.headshots > df.kills)).groupby(df.game_id).transform("any")
    kills_per_round = df.groupby("game_id").kills.transform("sum") / df.rounds
    df = df[~bad & (df.rounds >= 13) & kills_per_round.between(4.5, 9)]
    return df.sort_values(["start_date", "match_id", "map_number"], kind="stable").reset_index(drop=True)


def player_state(df, maps, ts):
    """Each player's form *after* each match (windows include that match's maps), and opp_elo_l20: the mean
    pre-match Elo of the opponents behind it (ts = team_state)."""
    opp = ts[["team_id", "start_date", "elo"]].rename(columns={"team_id": "opp_id", "elo": "opp_elo"})
    df = df.assign(opp_elo=pd.merge_asof(df[["start_date", "opp_id"]], opp, on="start_date", by="opp_id",
                                         allow_exact_matches=False).opp_elo.fillna(1500.0).to_numpy())
    g = df.groupby("player_id")
    st = df[["player_id", "start_date"]].copy()
    for col, n in [("kills", 20), ("kills", 5), ("headshots", 20), ("rounds", 20), ("adr", 20), ("kast", 20),
                   ("opp_elo", 20)]:
        st[f"{col}_l{n}"] = g[col].transform(lambda s: s.rolling(n, min_periods=min(n, 5)).mean())
    st["hs_pct"] = st.headshots_l20 / st.kills_l20
    st["maps_played"] = np.minimum(g.cumcount() + 1, 100)  # capped: a thin-history flag, not a calendar proxy
    for m in maps:  # all-time kills and rounds on each map
        on = df.map_name == m
        st[f"k_{m}"] = df.kills.where(on, 0).groupby(df.player_id).cumsum()
        st[f"r_{m}"] = df.rounds.where(on, 0).fillna(0).groupby(df.player_id).cumsum()
    return st.drop_duplicates(["player_id", "start_date"], keep="last")


def team_state(df, maps):
    """Each team's map Elo and count of each map in its last POOL maps, *after* each match."""
    rating, last_team, rows = {}, {}, []
    roster = df.groupby(["game_id", "team_id"], sort=False).player_id.agg(list)
    for r in df.drop_duplicates("game_id").itertuples():
        for t in (r.team_id, r.opp_id):  # a new team id (e.g. a roster moving org) starts from its players' old teams
            if t not in rating:
                prev = [rating[last_team[p]] for p in roster.get((r.game_id, t), []) if last_team.get(p) in rating]
                rating[t] = float(np.mean(prev)) if prev else 1500.0
        a, b = rating[r.team_id], rating[r.opp_id]
        if r.win == r.win:  # skip maps without a recorded winner
            d = 32 * (r.win - 1 / (1 + 10 ** ((b - a) / 400)))
            a, b = a + d, b - d
        rating[r.team_id], rating[r.opp_id] = a, b
        for t in (r.team_id, r.opp_id):
            last_team.update(dict.fromkeys(roster.get((r.game_id, t), []), t))
        rows += [(r.team_id, r.start_date, r.map_name, a), (r.opp_id, r.start_date, r.map_name, b)]
    ts = pd.DataFrame(rows, columns=["team_id", "start_date", "map_name", "elo"])
    for m in maps:
        ts[f"pool_{m}"] = (ts.map_name == m).astype(float).groupby(ts.team_id).transform(
            lambda s: s.rolling(POOL, min_periods=1).sum())
    return ts.drop(columns="map_name").drop_duplicates(["team_id", "start_date"], keep="last")


def features(q, df):
    """Features for queries (player_id, team_id, opp_id, start_date) from state strictly before start_date."""
    maps = sorted(df.map_name.dropna().unique())
    ts = team_state(df, maps)
    ps = player_state(df, maps, ts)
    opp = ts.rename(columns=lambda c: {"team_id": "opp_id", "elo": "elo_opp"}.get(c, c.replace("pool_", "opp_pool_")))
    f = q.sort_values("start_date", kind="stable")
    for right, by in ((ps, "player_id"), (ts, "team_id"), (opp, "opp_id")):
        f = pd.merge_asof(f, right, on="start_date", by=by, allow_exact_matches=False)
    f["kpr"] = f.kills_l20 / f.rounds_l20

    # Expected KPR on the maps these two teams tend to play, relative to the player's overall KPR.
    col = lambda p: np.nan_to_num(f[[f"{p}{m}" for m in maps]].to_numpy(float))  # noqa: E731
    w = col("pool_") + col("opp_pool_")
    kpr = f.kpr.to_numpy()[:, None]
    map_kpr = (col("k_") + PRIOR_ROUNDS * kpr) / (col("r_") + PRIOR_ROUNDS)
    with np.errstate(invalid="ignore", divide="ignore"):
        f["map_factor"] = np.where(w.sum(1) > 0, (w * map_kpr).sum(1) / w.sum(1) / kpr[:, 0], 1.0)
    f["elo_opp"] = f.elo_opp.fillna(1500.0)
    f["win_prob"] = 1 / (1 + 10 ** ((f.elo_opp - f.elo.fillna(1500.0)) / 400))
    # Is today's opponent tougher than the ones behind the player's 20-map averages? Without it, a player who
    # farmed weaker teams looks better than they are against a strong one. Unlike elo_opp it doesn't drift: top-tier
    # Elo inflated from about 1550 in 2023 to 1900 in 2026, so the model came to under-project tier-1 players.
    f["sched_gap"] = f.elo_opp - f.opp_elo_l20
    return f


def series(df):
    """(match, player) rows for best-of-3/5 series where the player played maps 1 and 2."""
    d = df[(df.bo_type >= 3) & df.map_number.isin([1, 2])]
    s = d.groupby(["match_id", "player_id"]).agg(
        start_date=("start_date", "first"), team_id=("team_id", "first"), opp_id=("opp_id", "first"),
        n=("map_number", "nunique"), kills_m12=("kills", "sum"), headshots_m12=("headshots", "sum"))
    return s[s.n == 2].drop(columns="n").reset_index()


MODEL = "gbm"  # "gbm": gradient-boosted trees (default); "tree": one decision tree
GRIDS = {  # settings tried on the validation slice; the lowest MAE wins
    "tree": [dict(max_depth=d, min_samples_leaf=n) for d in (3, 4, 5, 6, 8, 10) for n in (25, 50, 100, 200, 400)],
    "gbm": [dict(loss=loss, learning_rate=lr, max_leaf_nodes=n) for loss in ("squared_error", "poisson")
            for lr in (0.05, 0.1) for n in (15, 31)],
}


def fit(X, y, kind, params):
    if kind == "tree":
        return DecisionTreeRegressor(random_state=0, **params).fit(X, y)
    # Early stopping holds out 10% of the rows it is given and stops adding trees once they stop helping.
    return HistGradientBoostingRegressor(max_iter=1000, early_stopping=True, random_state=0, **params).fit(X, y)


def fit_models(feats, kind=MODEL, verbose=True, params=None):
    """Per stat: tune on a time split (or reuse `params`, {stat: settings}), score the newest 15%, then refit
    on all rows so the saved model has seen the most recent matches. Returns {stat: bundle}."""
    be, bundles = american_to_prob(DEFAULT_ODDS), {}
    for stat in STATS:
        y = f"{stat}_m12"
        s = feats.dropna(subset=NEED[stat] + [y])
        n = len(s)
        tr, va, te = s[: int(0.7 * n)], s[int(0.7 * n): int(0.85 * n)], s[int(0.85 * n):]

        def mae(m, d):
            return np.abs(m.predict(d[FEATURES]) - d[y]).mean()

        best = params[stat] if params else min(GRIDS[kind], key=lambda p: mae(fit(tr[FEATURES], tr[y], kind, p), va))
        model = fit(tr[FEATURES], tr[y], kind, best)
        p = np.maximum(model.predict(va[FEATURES]), 1e-9)
        z = ((va[y] - p) / np.sqrt(p)).to_numpy()
        z = z - z.mean()  # keep the spread, drop any level shift between the train and validation periods

        # Honest check on the untouched last 15%: does the model beat "2 x 20-map average"?
        pred, base, actual = model.predict(te[FEATURES]), 2 * te[f"{stat}_l20"].to_numpy(), te[y].to_numpy()
        line = np.floor(base) + 0.5  # stand-in line; real books are sharper than this
        po, pu = probs(pred, line, z)
        bet_o, bet_u = po > be, pu > be
        bets = bet_o.sum() + bet_u.sum()
        metrics = dict(model=kind, stat=stat, test_mae=np.abs(pred - actual).mean(),
                       avg20_mae=np.abs(base - actual).mean(), bets=bets,
                       hit=((actual[bet_o] > line[bet_o]).sum() + (actual[bet_u] < line[bet_u]).sum()) / max(bets, 1),
                       brier=np.mean((po - (actual > line)) ** 2))
        if verbose:
            print(f"\n{stat} ({kind}): {n} player-series | train {len(tr)} | validate {len(va)} | test {len(te)} "
                  f"({te.start_date.iloc[0]:%Y-%m-%d} to {te.start_date.iloc[-1]:%Y-%m-%d})\n   {best}\n"
                  f"   test MAE: model {metrics['test_mae']:.2f} vs 2 x 20-map avg {metrics['avg20_mae']:.2f}\n"
                  f"   stand-in-line bets {bets}/{len(te)}, hit {metrics['hit']:.1%} (breakeven {be:.1%}; "
                  f"optimistic, real lines are sharper) | Brier {metrics['brier']:.3f} vs 0.250")
            sample = te.sample(min(3000, len(te)), random_state=0)
            imp = permutation_importance(model, sample[FEATURES], sample[y], n_repeats=3, random_state=0,
                                         scoring="neg_mean_absolute_error").importances_mean
            print("   top features (MAE rise when shuffled): " +
                  ", ".join(f"{name} {v:.2f}" for v, name in sorted(zip(imp, FEATURES), reverse=True)[:5]))

        # ponytail: z comes from the 70%-data model, so production intervals are slightly wide.
        bundles[stat] = {"model": fit(s[FEATURES], s[y], kind, best), "z": np.sort(z), "metrics": metrics,
                         "params": best}
    return bundles


def train(maps_csv=MAPS_CSV, kind=MODEL):
    df = load(maps_csv)
    for stat, bundle in fit_models(features(series(df), df), kind).items():
        joblib.dump(bundle, f"data/model_{stat}.joblib")
    print("saved data/model_kills.joblib, data/model_headshots.joblib")


def compare(maps_csv=MAPS_CSV):
    """Decision tree vs gradient boosting on the same time split (nothing is saved)."""
    df = load(maps_csv)
    feats = features(series(df), df)
    rows = [b["metrics"] for kind in ("tree", "gbm") for b in fit_models(feats, kind, verbose=False).values()]
    print(pd.DataFrame(rows).set_index(["stat", "model"]).round(3).to_string())


def key(x):
    return re.sub(r"[^a-z0-9]", "", str(x).lower())


def team_key(x):
    return key(re.sub(r"\b(team|esports?|gaming|club|cs2?|br)\b", " ", str(x).lower()))


def resolve(lines, df):
    """Match line player / team / opponent names to bo3.gg ids (NaN / -1 when unknown)."""
    recent = df[df.start_date > df.start_date.max() - pd.Timedelta(days=180)]
    players = recent.drop_duplicates("player_id", keep="last")  # ordered by last appearance
    players = players.assign(nick=players.nickname.map(key))
    teams = recent.drop_duplicates("team_id", keep="last")
    team_ids = dict(zip(teams.team_name.map(team_key), teams.team_id))  # exact only: fuzzy team names mis-join

    out = lines.assign(player_id=np.nan, team_id=-1, opp_id=-1, matched="")
    for i, ln in lines.iterrows():
        nick, tid = key(ln.player), team_ids.get(team_key(ln.team), -1)
        on_team = players[players.team_id == tid]
        exact = players[players.nick == nick]
        close = difflib.get_close_matches(nick, list(on_team.nick), 1, 0.6)
        # Exact name on the team, then exact name anywhere (transfers), then a close name on the team
        # (Sleeper "MaxOff" = bo3 "max"). Always the most recently active candidate.
        p = (exact[exact.team_id == tid] if (exact.team_id == tid).any() else exact if len(exact) else
             on_team[on_team.nick == close[0]] if close else exact)
        if len(p):
            p = p.iloc[-1]
            # Show the bo3 team when it differs, so a namesake on another team is easy to spot.
            shown = p.nickname if p.team_id == tid else f"{p.nickname} ({p.team_name})"
            out.loc[i, ["player_id", "team_id", "opp_id", "matched"]] = [
                p.player_id, tid if tid != -1 else p.team_id, team_ids.get(team_key(ln.opponent), -1), shown]
    return out


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def cal_x(d):
    """Calibration input: the model's logit P(over | no push)."""
    return logit(d.p_over / (d.p_over + d.p_under))[:, None]


def fit_calibration(graded):
    """{stat: P(over) against real lines = sigmoid(a + b * model logit)}, fitted on graded past lines, for each
    stat with MIN_CAL_LINES decided lines (fewer give wild fits; that stat stays uncalibrated).

    The books' line is the strongest single predictor there is, so a model that ignores it is overconfident
    whenever it disagrees with it: b < 1 measures by how much, a measures a lean toward one side."""
    d = graded[graded.result.isin(["OVER", "UNDER"]) & graded.p_over.notna()]
    return {stat: LogisticRegression().fit(cal_x(g), g.result == "OVER") for stat, g in d.groupby("stat")
            if len(g) >= MIN_CAL_LINES and g.result.nunique() == 2}


def describe_calibration(cal):
    return (" | ".join(f"{stat} a={lr.intercept_[0]:+.2f} b={lr.coef_[0, 0]:.2f}" for stat, lr in cal.items())
            + " (b = how much of the model's disagreement with the line is real: 1 all of it, 0 none)")


def calibrated(out, cal):
    """Replace P(over/under) with the calibrated ones for each stat in `cal`; the push probability is unchanged."""
    for stat, lr in (cal or {}).items():
        m = out.p_over.notna() & (out.stat == stat)
        if m.any():
            keep = (out.p_over + out.p_under)[m]
            p = lr.predict_proba(cal_x(out[m]))[:, 1]
            out.loc[m, "p_over"], out.loc[m, "p_under"] = p * keep, (1 - p) * keep
    return out


def picks(out):
    """Edge (probability minus the price's breakeven) and OVER / UNDER / PASS / NO DATA."""
    be_over, be_under = out.over_odds.map(american_to_prob), out.under_odds.map(american_to_prob)
    out["edge"] = np.maximum(out.p_over - be_over, out.p_under - be_under)
    out["pick"] = np.select([out.p_over.isna(), out.p_over > be_over, out.p_under > be_under],
                            ["NO DATA", "OVER", "UNDER"], "PASS")
    return out


def score(lines, df, bundles, cal=None):
    """Projection, P(over/under), edge and pick for maps 1-2 kill/headshot lines (start_time = as-of time).
    `cal` (from fit_calibration) maps the model's P(over) onto what happened against real lines."""
    lines = lines[lines.stat.isin(STATS) & (lines.maps.astype(str) == "1-2")].reset_index(drop=True)
    lines["start_date"] = pd.to_datetime(lines.start_time, utc=True, format="ISO8601").fillna(pd.Timestamp.now(tz="UTC"))
    lines = resolve(lines, df).reset_index(names="line_idx")
    # Unmatched opponent names get NO DATA rather than being scored as an average team.
    known = lines[lines.player_id.notna() & (lines.opp_id != -1)].astype({"player_id": int, "team_id": int, "opp_id": int})
    f = features(known[["line_idx", "player_id", "team_id", "opp_id", "start_date"]], df)
    out = lines.merge(f[["line_idx"] + FEATURES], on="line_idx", how="left")
    if "market_win_prob" in out:  # the market's win chance (lines.py --pickfinder) is sharper than our Elo's
        out["elo_win_prob"] = out.win_prob
        out["win_prob"] = pd.to_numeric(out.market_win_prob, errors="coerce").fillna(out.win_prob)
    out["proj"] = out["p_over"] = out["p_under"] = np.nan
    for stat, bundle in bundles.items():
        m = (out.stat == stat) & out[NEED[stat]].notna().all(axis=1)
        if m.any():
            out.loc[m, "proj"] = bundle["model"].predict(out.loc[m, FEATURES])
            out.loc[m, "p_over"], out.loc[m, "p_under"] = probs(out.loc[m, "proj"], out.loc[m, "line"], bundle["z"])
    for side in ("over", "under"):
        odds = out[f"{side}_odds"] if f"{side}_odds" in out else pd.Series(np.nan, index=out.index)
        out[f"{side}_odds"] = pd.to_numeric(odds, errors="coerce").fillna(DEFAULT_ODDS)
    return picks(calibrated(out, cal))


def predict(lines_csv=LINES_CSV, maps_csv=MAPS_CSV):
    df = load(maps_csv)
    bundles = {stat: joblib.load(f"data/model_{stat}.joblib") for stat in STATS}
    if any(list(b["model"].feature_names_in_) != FEATURES for b in bundles.values()):
        return print("The saved models predate the current features: run python -m cs2props.model train first.")
    saved = joblib.load(CAL_JOBLIB) if os.path.exists(CAL_JOBLIB) else {}
    # Only a calibration of these features and this kind of model applies; any other is stale.
    same = (saved.get("features"), saved.get("kind")) == (FEATURES, bundles["kills"]["metrics"]["model"])
    cal = saved.get("cal") if same else None
    out = score(pd.read_csv(lines_csv), df, bundles, cal).sort_values("edge", ascending=False)
    cols = ["book", "player", "matched", "team", "opponent", "stat", "line", "proj", "elo_win_prob", "market_win_prob",
            "p_over", "p_under", "edge", "pick"]
    out.to_csv(PICKS_CSV, index=False)
    print(out[[c for c in cols if c in out]].round(3).to_string(index=False))
    print(f"\n{(out.pick == 'NO DATA').sum()} of {len(out)} lines had no usable history (check 'matched').")
    print(f"P(over/under) calibrated on real past lines ({CAL_JOBLIB}): {describe_calibration(cal)}" if cal else
          f"P(over/under) NOT calibrated on real lines: run backtest first (it writes {CAL_JOBLIB}).")


def profit(odds):
    """Profit per 1 unit staked at American odds when the pick wins."""
    return np.where(odds > 0, odds / 100, 100 / np.abs(odds))


def bets(out):
    """The OVER/UNDER picks among graded lines, with units won at each line's own price."""
    bet = out[out.pick.isin(["OVER", "UNDER"])].copy()
    odds = np.where(bet.pick == "OVER", bet.over_odds, bet.under_odds)
    bet["units"] = np.select([bet.result == bet.pick, bet.result == "PUSH"], [profit(odds), 0.0], -1.0)
    bet["breakeven"] = pd.Series(odds, index=bet.index).map(american_to_prob)
    bet["won"], bet["push"] = bet.result == bet.pick, bet.result == "PUSH"
    bet["series_key"] = bet.player_id.astype(str) + bet.stat + bet.match_start.astype(str)
    return bet


def summary(g):
    decided = g[~g.push]
    hit = decided.won.mean() if len(decided) else np.nan
    # Copies of one line at several books share one outcome, so the band counts player-series, not rows.
    n = decided.series_key.nunique()
    ci = 1.96 * np.sqrt(hit * (1 - hit) / max(n, 1))
    return pd.Series({"bets": len(g), "series": n, "hit": hit, "+/-95%": ci, "breakeven": g.breakeven.mean(),
                      "units": g.units.sum(), "roi": g.units.sum() / max(len(g), 1)})


def report(out, title):
    """Hit rate, units and ROI of the picks in graded lines `out`, split several ways, and a calibration table."""
    bet = bets(out)
    print(f"\n===== {title}: {len(out)} graded lines from {out.start_date.min():%Y-%m-%d} to "
          f"{out.start_date.max():%Y-%m-%d}; {len(bet)} bets, {(out.result == 'PUSH').sum()} pushes =====")
    if len(bet):
        print("\nAll bets:\n" + summary(bet).round(3).to_string())
        bet["edge_bucket"] = pd.cut(bet.edge, [0, 0.05, 0.10, 1], labels=["0-5%", "5-10%", "10%+"])
        for col in ("edge_bucket", "stat", "pick", "book"):
            print(f"\nBy {col}:\n" + bet.groupby(col, observed=True).apply(summary, include_groups=False).round(3).to_string())
    cal = out[out.result != "PUSH"].assign(bin=lambda d: pd.cut(d.p_over, [0, .3, .4, .5, .6, .7, 1]))
    print("\nCalibration (predicted P(over) vs actual over rate):\n" + cal.groupby("bin", observed=True).agg(
        lines=("p_over", "size"), predicted=("p_over", "mean"),
        actual=("result", lambda r: (r == "OVER").mean())).round(3).to_string())
    return bet


def backtest(props_csv=PROPS_CSV, kind=MODEL, maps_csv=MAPS_CSV):
    """Grade the model on real past lines (PickFinder prop history, or data/lines_history.csv).

    Each line is first matched to its own bo3.gg series: same player and opponent, starting within
    90 minutes (book and bo3.gg times differ by minutes; same-day rematches are 2+ hours apart). Features
    are then taken strictly before the earlier of the two start times, and each calendar month's series are
    scored by trees refit on matches that started at least a day before that month, so nothing graded leaks
    into a prediction. The calibration against real lines is walk-forward too: a month only uses lines that
    were settled before it began.
    """
    df = load(maps_csv)
    props = pd.read_csv(props_csv)
    props = props[props.stat.isin(STATS) & (props.maps.astype(str) == "1-2")].reset_index(drop=True)
    props["start_date"] = pd.to_datetime(props.start_time, utc=True, format="ISO8601")
    props = resolve(props.dropna(subset=["start_date"]).reset_index(drop=True), df)
    res = series(df)[["player_id", "opp_id", "start_date", "kills_m12", "headshots_m12"]]
    res = res.rename(columns={"start_date": "match_start"}).astype({"player_id": float, "opp_id": int})
    props = pd.merge_asof(props.sort_values("start_date"), res.sort_values("match_start"), left_on="start_date",
                          right_on="match_start", by=["player_id", "opp_id"], direction="nearest",
                          tolerance=pd.Timedelta(minutes=90))
    props = props[props.match_start.notna()].copy()
    props["start_time"] = props.start_date.where(props.start_date < props.match_start, props.match_start)
    if "scraped_at" in props:  # our archive: each book's last snapshot before the match is its closing line
        props = props[pd.to_datetime(props.scraped_at, utc=True, format="ISO8601") < props.start_time]
        props = props.sort_values("scraped_at")
    # One line per book per series (PickFinder can list a rescheduled match twice, a day apart).
    props = props.drop_duplicates(["book", "player_id", "stat", "maps", "match_start"], keep="last")
    if props.empty:
        return print("no gradable lines: none matched a bo3.gg series of that player vs that opponent "
                     "(run history first; the match may not be finished)")

    feats = features(series(df), df)
    # Walk forward like a regular `train` would: each month's series are scored by trees refit on every match
    # that started a day or more before that calendar month, so all had finished before its first line (settings
    # tuned once, on the first month). One model fitted before the first line drifted: its bias grew from -0.1 to
    # +1.0 kills over three months.
    props["month"] = props.match_start.dt.strftime("%Y-%m")
    known = lambda m: pd.Timestamp(f"{m}-01", tz="UTC") - pd.Timedelta(days=1)  # noqa: E731
    parts, params = [], None
    for month, chunk in props.groupby("month"):
        bundles = fit_models(feats[feats.start_date < known(month)], kind, verbose=False, params=params)
        params = {stat: b["params"] for stat, b in bundles.items()}
        lines = chunk.drop(columns=["start_date", "player_id", "team_id", "opp_id", "matched"])
        part = score(lines, df, bundles)
        if "market_win_prob" in lines and lines.market_win_prob.notna().any():  # the same lines on our Elo alone
            part["proj_elo"] = score(lines.drop(columns="market_win_prob"), df, bundles).proj.to_numpy()
        parts.append(part)
    out = pd.concat(parts, ignore_index=True)

    bo3 = pd.Series(np.where(out.stat == "kills", out.kills_m12, out.headshots_m12), index=out.index)
    if "actual" in out:  # the file's result is what settled; bo3.gg is a cross-check
        both = out.actual.notna()
        print(f"actual results: source file agrees with bo3.gg on {(out.actual[both] == bo3[both]).mean():.1%} "
              f"of {both.sum()} lines")
        out["actual"] = out.actual.fillna(bo3)
    else:
        out["actual"] = bo3
    out = out[out.p_over.notna()].copy()
    if out.empty:
        return print("no gradable lines: the matched players have too little history before the line")
    out["result"] = np.select([out.actual > out.line, out.actual < out.line], ["OVER", "UNDER"], "PUSH")
    bet = report(out, f"{kind} model, refit monthly")
    over_rate = (out.result == "OVER")[out.result != "PUSH"].mean()
    print(f"\nBaselines: always OVER hits {over_rate:.1%}, always UNDER hits {1 - over_rate:.1%}")
    if "proj_elo" in out:  # did the market's win chance make projections better than our Elo's?
        mk = out.proj_elo.notna() & out.market_win_prob.notna()
        err = lambda p: (out.actual - p)[mk]  # noqa: E731
        print(f"Market win chance on {mk.sum()} graded lines: projection error {err(out.proj).abs().mean():.2f} "
              f"(bias {err(out.proj).mean():+.2f}) vs {err(out.proj_elo).abs().mean():.2f} "
              f"(bias {err(out.proj_elo).mean():+.2f}) with our Elo's")
    # Line shopping, no model: where books post different lines on one series, take the side the gap favours.
    # Our archive's lines are compared within one scrape only (a pulled line next to another book's later,
    # moved one was never on offer). PickFinder's lines carry no time, so there the books may not overlap.
    timed = "scraped_at" in out
    key = ["player_id", "stat", "match_start"] + (["scraped_at"] if timed else [])
    lo, hi = (out.groupby(key).line.transform(f) for f in ("min", "max"))
    o, u = out[(hi > lo) & (out.line == lo)], out[(hi > lo) & (out.line == hi)]
    if len(o):
        print(f"Line shopping{'' if timed else ' (untimed lines)'}, {o.groupby(key).ngroups} series where books "
              f"disagree by {(hi - lo)[hi > lo].mean():.1f} on average: OVER at the lowest line hits "
              f"{(o.result == 'OVER')[o.result != 'PUSH'].mean():.1%} (breakeven "
              f"{o.over_odds.map(american_to_prob).mean():.1%}), UNDER at the highest hits "
              f"{(u.result == 'UNDER')[u.result != 'PUSH'].mean():.1%} (breakeven "
              f"{u.under_odds.map(american_to_prob).mean():.1%})")

    # Calibrated against the books, walk-forward: each month is calibrated only on lines settled before it began
    # (and only for the stats that had enough of them).
    cals = {m: fit_calibration(out[out.match_start < known(m)]) for m in sorted(out.month.unique())}
    cal_parts = [picks(calibrated(out[(out.month == m) & out.stat.isin(list(cal))].copy(), cal))
                 for m, cal in cals.items() if cal]
    if cal_parts:
        cal_out = pd.concat(cal_parts)
        same = summary(bets(out.loc[cal_out.index]))
        report(cal_out, "Calibrated on earlier months' lines")
        print(f"\nUncalibrated on these same lines: {same.bets:.0f} bets, hit {same.hit:.3f}, roi {same.roi:+.3f}")
        out = out.join(cal_out[["p_over", "p_under", "edge", "pick"]].add_suffix("_cal"))
    else:
        print(f"\n(Calibrated report needs {MIN_CAL_LINES}+ graded lines of a stat from before a month.)")
    cal = fit_calibration(out)
    if cal:  # a small file must not replace a well-founded calibration
        lines = out[out.result.isin(["OVER", "UNDER"])].stat.value_counts()
        joblib.dump({"cal": cal, "features": FEATURES, "kind": kind, "lines": {s: int(lines[s]) for s in cal}},
                    CAL_JOBLIB)
        print(f"\nCalibration fitted on all graded lines, used by predict ({CAL_JOBLIB}):\n   "
              + describe_calibration(cal))
    else:
        print(f"\nCalibration not saved (no stat had {MIN_CAL_LINES} decided lines); {CAL_JOBLIB} unchanged.")
    out.merge(bet[["units"]], left_index=True, right_index=True, how="left").to_csv(BACKTEST_CSV, index=False)
    print(f"\nper-line results saved to {BACKTEST_CSV}")


if __name__ == "__main__":
    cmd, *args = sys.argv[1:] or ["train"]
    {"train": train, "predict": predict, "backtest": backtest, "compare": compare}[cmd](*args)
