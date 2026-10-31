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
import re
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeRegressor

MAPS_CSV, LINES_CSV, PICKS_CSV = "data/maps.csv", "data/lines.csv", "data/picks.csv"
PROPS_CSV, BACKTEST_CSV = "data/prop_history.csv", "data/backtest.csv"
STATS = ("kills", "headshots")
FEATURES = ["kills_l20", "kills_l5", "headshots_l20", "hs_pct", "kpr", "adr_l20", "kast_l20", "rounds_l20",
            "maps_played", "map_factor", "win_prob", "elo_opp"]
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


def player_state(df, maps):
    """Each player's form *after* each match (windows include that match's maps)."""
    g = df.groupby("player_id")
    st = df[["player_id", "start_date"]].copy()
    for col, n in [("kills", 20), ("kills", 5), ("headshots", 20), ("rounds", 20), ("adr", 20), ("kast", 20)]:
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
    ps, ts = player_state(df, maps), team_state(df, maps)
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


def fit_models(feats, kind=MODEL, verbose=True):
    """Per stat: tune on a time split, score the newest 15%, refit on the rest. Returns {stat: bundle}."""
    be, bundles = american_to_prob(DEFAULT_ODDS), {}
    for stat in STATS:
        y = f"{stat}_m12"
        s = feats.dropna(subset=NEED[stat] + [y])
        n = len(s)
        tr, va, te = s[: int(0.7 * n)], s[int(0.7 * n): int(0.85 * n)], s[int(0.85 * n):]

        def mae(m, d):
            return np.abs(m.predict(d[FEATURES]) - d[y]).mean()

        params = min(GRIDS[kind], key=lambda p: mae(fit(tr[FEATURES], tr[y], kind, p), va))
        model = fit(tr[FEATURES], tr[y], kind, params)
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
                  f"({te.start_date.iloc[0]:%Y-%m-%d} to {te.start_date.iloc[-1]:%Y-%m-%d})\n   {params}\n"
                  f"   test MAE: model {metrics['test_mae']:.2f} vs 2 x 20-map avg {metrics['avg20_mae']:.2f}\n"
                  f"   stand-in-line bets {bets}/{len(te)}, hit {metrics['hit']:.1%} (breakeven {be:.1%}; "
                  f"optimistic, real lines are sharper) | Brier {metrics['brier']:.3f} vs 0.250")
            sample = te.sample(min(3000, len(te)), random_state=0)
            imp = permutation_importance(model, sample[FEATURES], sample[y], n_repeats=3, random_state=0,
                                         scoring="neg_mean_absolute_error").importances_mean
            print("   top features (MAE rise when shuffled): " +
                  ", ".join(f"{name} {v:.2f}" for v, name in sorted(zip(imp, FEATURES), reverse=True)[:5]))

        # ponytail: z comes from the 70%-data model, so production intervals are slightly wide.
        both = pd.concat([tr, va])
        bundles[stat] = {"model": fit(both[FEATURES], both[y], kind, params), "z": np.sort(z), "metrics": metrics}
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


def score(lines, df, bundles):
    """Projection, P(over/under), edge and pick for maps 1-2 kill/headshot lines (start_time = as-of time)."""
    lines = lines[lines.stat.isin(STATS) & (lines.maps.astype(str) == "1-2")].reset_index(drop=True)
    lines["start_date"] = pd.to_datetime(lines.start_time, utc=True, format="ISO8601").fillna(pd.Timestamp.now(tz="UTC"))
    lines = resolve(lines, df).reset_index(names="line_idx")
    # Unmatched opponent names get NO DATA rather than being scored as an average team.
    known = lines[lines.player_id.notna() & (lines.opp_id != -1)].astype({"player_id": int, "team_id": int, "opp_id": int})
    f = features(known[["line_idx", "player_id", "team_id", "opp_id", "start_date"]], df)
    out = lines.merge(f[["line_idx"] + FEATURES], on="line_idx", how="left")
    out["proj"] = out["p_over"] = out["p_under"] = np.nan
    for stat, bundle in bundles.items():
        m = (out.stat == stat) & out[NEED[stat]].notna().all(axis=1)
        if m.any():
            out.loc[m, "proj"] = bundle["model"].predict(out.loc[m, FEATURES])
            out.loc[m, "p_over"], out.loc[m, "p_under"] = probs(out.loc[m, "proj"], out.loc[m, "line"], bundle["z"])
    for side in ("over", "under"):
        odds = out[f"{side}_odds"] if f"{side}_odds" in out else pd.Series(np.nan, index=out.index)
        out[f"{side}_odds"] = pd.to_numeric(odds, errors="coerce").fillna(DEFAULT_ODDS)
    be_over, be_under = out.over_odds.map(american_to_prob), out.under_odds.map(american_to_prob)
    out["edge"] = np.maximum(out.p_over - be_over, out.p_under - be_under)
    out["pick"] = np.select([out.p_over.isna(), out.p_over > be_over, out.p_under > be_under],
                            ["NO DATA", "OVER", "UNDER"], "PASS")
    return out


def predict(lines_csv=LINES_CSV, maps_csv=MAPS_CSV):
    df = load(maps_csv)
    bundles = {stat: joblib.load(f"data/model_{stat}.joblib") for stat in STATS}
    out = score(pd.read_csv(lines_csv), df, bundles).sort_values("edge", ascending=False)
    cols = ["book", "player", "matched", "team", "opponent", "stat", "line", "proj", "p_over", "p_under", "edge", "pick"]
    out.to_csv(PICKS_CSV, index=False)
    print(out[[c for c in cols if c in out]].round(3).to_string(index=False))
    print(f"\n{(out.pick == 'NO DATA').sum()} of {len(out)} lines had no usable history (check 'matched').")


def profit(odds):
    """Profit per 1 unit staked at American odds when the pick wins."""
    return np.where(odds > 0, odds / 100, 100 / np.abs(odds))


def backtest(props_csv=PROPS_CSV, kind=MODEL, maps_csv=MAPS_CSV):
    """Grade the model on real past lines (PickFinder prop history, or data/lines_history.csv).

    Each line is first matched to its own bo3.gg series: same player and opponent, starting within
    90 minutes (book and bo3.gg times differ by minutes; same-day rematches are 2+ hours apart). Features
    are then taken strictly before the earlier of the two start times, and the trees are refit only on
    matches before the first graded line, so the graded match never leaks into its own prediction.
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
        props = props.sort_values("scraped_at").drop_duplicates(["book", "player_id", "stat", "maps", "match_start"],
                                                                keep="last")
    if props.empty:
        return print("no gradable lines: none matched a bo3.gg series of that player vs that opponent "
                     "(run history first; the match may not be finished)")

    feats = features(series(df), df)
    cutoff = props.start_time.min()
    # ponytail: one cutoff; a monthly walk-forward refit would use more training data for later lines.
    bundles = fit_models(feats[feats.start_date < cutoff], kind, verbose=False)
    out = score(props.drop(columns=["start_date", "player_id", "team_id", "opp_id", "matched"]), df, bundles)

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
    bet = out[out.pick.isin(["OVER", "UNDER"])].copy()
    odds = np.where(bet.pick == "OVER", bet.over_odds, bet.under_odds)
    bet["units"] = np.select([bet.result == bet.pick, bet.result == "PUSH"], [profit(odds), 0.0], -1.0)
    bet["breakeven"] = pd.Series(odds, index=bet.index).map(american_to_prob)
    bet["won"], bet["push"] = bet.result == bet.pick, bet.result == "PUSH"
    bet["series_key"] = bet.player_id.astype(str) + bet.stat + bet.match_start.astype(str)
    out.merge(bet[["units"]], left_index=True, right_index=True, how="left").to_csv(BACKTEST_CSV, index=False)

    def summary(g):
        decided = g[~g.push]
        hit = decided.won.mean() if len(decided) else np.nan
        # Copies of one line at several books share one outcome, so the band counts player-series, not rows.
        n = decided.series_key.nunique()
        ci = 1.96 * np.sqrt(hit * (1 - hit) / max(n, 1))
        return pd.Series({"bets": len(g), "series": n, "hit": hit, "+/-95%": ci, "breakeven": g.breakeven.mean(),
                          "units": g.units.sum(), "roi": g.units.sum() / max(len(g), 1)})

    print(f"\n{len(out)} graded lines from {out.start_date.min():%Y-%m-%d} to {out.start_date.max():%Y-%m-%d} "
          f"({kind} models trained on matches before {cutoff:%Y-%m-%d}); {len(bet)} bets, {(out.result == 'PUSH').sum()} pushes")
    print("\nAll bets:\n" + summary(bet).round(3).to_string())
    bet["edge_bucket"] = pd.cut(bet.edge, [0, 0.05, 0.10, 1], labels=["0-5%", "5-10%", "10%+"])
    for col in ("edge_bucket", "stat", "pick", "book"):
        print(f"\nBy {col}:\n" + bet.groupby(col, observed=True).apply(summary, include_groups=False).round(3).to_string())
    over_rate = (out.result == "OVER")[out.result != "PUSH"].mean()
    print(f"\nBaselines: always OVER hits {over_rate:.1%}, always UNDER hits {1 - over_rate:.1%}")
    cal = out[out.result != "PUSH"].assign(bin=lambda d: pd.cut(d.p_over, [0, .3, .4, .5, .6, .7, 1]))
    print("\nCalibration (predicted P(over) vs actual over rate):\n" + cal.groupby("bin", observed=True).agg(
        lines=("p_over", "size"), predicted=("p_over", "mean"),
        actual=("result", lambda r: (r == "OVER").mean())).round(3).to_string())
    print(f"\nper-line results saved to {BACKTEST_CSV}")


if __name__ == "__main__":
    cmd, *args = sys.argv[1:] or ["train"]
    {"train": train, "predict": predict, "backtest": backtest, "compare": compare}[cmd](*args)
