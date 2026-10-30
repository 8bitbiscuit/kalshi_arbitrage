"""Features, training and prediction for CS2 "Maps 1-2" kill / headshot props.

    python -m cs2props.model train     # data/maps.csv  -> data/model_{kills,headshots}.joblib
    python -m cs2props.model predict   # data/lines.csv -> data/picks.csv

The tree predicts the stat itself (e.g. maps 1-2 kills). There are no historical prop lines to label
"over/under" with, so P(over line) comes from the tree's projection times its held-out error ratios.
Any line, from any book, can then be scored without retraining.
"""
import difflib
import re
import sys
from itertools import product

import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

MAPS_CSV, LINES_CSV, PICKS_CSV = "data/maps.csv", "data/lines.csv", "data/picks.csv"
STATS = ("kills", "headshots")
FEATURES = ["kills_l20", "kills_l5", "headshots_l20", "hs_pct", "kpr", "adr_l20", "kast_l20", "rounds_l20",
            "maps_played", "map_factor", "win_prob", "elo_opp"]
PRIOR_ROUNDS = 100   # shrink a player's per-map KPR toward their recent KPR by this many rounds
POOL = 15            # a team's last 15 maps stand in for the unknown veto
DEFAULT_ODDS = -119  # lines without a price (PrizePicks): breakeven 54.3%, about a 5/6-pick flex entry


def american_to_prob(odds):
    return 100 / (odds + 100) if odds > 0 else -odds / (-odds + 100)


def probs(pred, line, ratios):
    """(P(over), P(under)): the integer stat is modelled as round(projection x held-out actual/pred ratio)."""
    r, pred, line = np.sort(ratios), np.maximum(np.asarray(pred, float), 1e-9), np.asarray(line, float)
    over = 1 - np.searchsorted(r, (np.floor(line) + 0.5) / pred, side="right") / len(r)
    under = np.searchsorted(r, (np.ceil(line) - 0.5) / pred, side="left") / len(r)
    return over, under


def load(path=MAPS_CSV):
    df = pd.read_csv(path)
    df["start_date"] = pd.to_datetime(df.start_date, utc=True, format="ISO8601")
    df = df.drop_duplicates(["game_id", "player_id"])
    return df.sort_values(["start_date", "match_id", "map_number"], kind="stable").reset_index(drop=True)


def player_state(df, maps):
    """Each player's form *after* each match (windows include that match's maps)."""
    g = df.groupby("player_id")
    st = df[["player_id", "start_date"]].copy()
    for col, n in [("kills", 20), ("kills", 5), ("headshots", 20), ("rounds", 20), ("adr", 20), ("kast", 20)]:
        st[f"{col}_l{n}"] = g[col].transform(lambda s: s.rolling(n, min_periods=min(n, 5)).mean())
    st["maps_played"] = g.cumcount() + 1
    for m in maps:  # all-time kills and rounds on each map
        on = df.map_name == m
        st[f"k_{m}"] = df.kills.where(on, 0).groupby(df.player_id).cumsum()
        st[f"r_{m}"] = df.rounds.where(on, 0).fillna(0).groupby(df.player_id).cumsum()
    return st.drop_duplicates(["player_id", "start_date"], keep="last")


def team_state(df, maps):
    """Each team's map Elo and count of each map in its last POOL maps, *after* each match."""
    rating, rows = {}, []
    for r in df.drop_duplicates("game_id").itertuples():
        a, b = rating.get(r.team_id, 1500.0), rating.get(r.opp_id, 1500.0)
        if r.win == r.win:  # skip maps without a recorded winner
            d = 32 * (r.win - 1 / (1 + 10 ** ((b - a) / 400)))
            a, b = a + d, b - d
        rating[r.team_id], rating[r.opp_id] = a, b
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
    f["hs_pct"] = f.headshots_l20 / f.kills_l20

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


def fit(X, y, depth, leaf):
    return DecisionTreeRegressor(max_depth=depth, min_samples_leaf=leaf, random_state=0).fit(X, y)


def train(maps_csv=MAPS_CSV):
    df = load(maps_csv)
    s = features(series(df), df).dropna(subset=FEATURES)
    n = len(s)
    tr, va, te = s[: int(0.7 * n)], s[int(0.7 * n): int(0.85 * n)], s[int(0.85 * n):]
    print(f"{n} player-series: train {len(tr)} | validate {len(va)} | test {len(te)} "
          f"(test = {te.start_date.iloc[0]:%Y-%m-%d} to {te.start_date.iloc[-1]:%Y-%m-%d})")
    be = american_to_prob(DEFAULT_ODDS)
    for stat in STATS:
        y = f"{stat}_m12"

        def mae(m, d):
            return np.abs(m.predict(d[FEATURES]) - d[y]).mean()

        depth, leaf = min(product((3, 4, 5, 6, 8, 10), (25, 50, 100, 200, 400)),
                          key=lambda p: mae(fit(tr[FEATURES], tr[y], *p), va))
        model = fit(tr[FEATURES], tr[y], depth, leaf)
        ratios = (va[y] / np.maximum(model.predict(va[FEATURES]), 1e-9)).to_numpy()

        # Honest check on the untouched last 15%: does the tree beat "2 x 20-map average"?
        pred, base, actual = model.predict(te[FEATURES]), 2 * te[f"{stat}_l20"].to_numpy(), te[y].to_numpy()
        line = np.floor(base) + 0.5  # stand-in line; real books are sharper than this
        po, pu = probs(pred, line, ratios)
        bet_o, bet_u = po > be, pu > be
        hits = (actual[bet_o] > line[bet_o]).sum() + (actual[bet_u] < line[bet_u]).sum()
        bets = bet_o.sum() + bet_u.sum()
        print(f"{stat}: depth={depth} leaf={leaf} | test MAE tree {np.abs(pred - actual).mean():.2f} vs "
              f"20-map avg {np.abs(base - actual).mean():.2f} | stand-in-line bets {bets}/{len(te)} hit "
              f"{hits / max(bets, 1):.1%} (breakeven {be:.1%}; optimistic, real lines are sharper) | "
              f"Brier {np.mean((po - (actual > line)) ** 2):.3f} vs 0.250")
        top = sorted(zip(model.feature_importances_, FEATURES), reverse=True)[:5]
        print("   top features: " + ", ".join(f"{name} {imp:.2f}" for imp, name in top))

        # ponytail: ratios come from the 70%-data model, so production intervals are slightly wide.
        both = pd.concat([tr, va])
        joblib.dump({"model": fit(both[FEATURES], both[y], depth, leaf), "ratios": np.sort(ratios)},
                    f"data/model_{stat}.joblib")
    print("saved data/model_kills.joblib, data/model_headshots.joblib")


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
    team_ids = dict(zip(teams.team_name.map(team_key), teams.team_id))

    def team(name):
        hit = difflib.get_close_matches(team_key(name), list(team_ids), 1, 0.8)
        return team_ids[hit[0]] if hit else -1

    out = lines.assign(player_id=np.nan, team_id=-1, opp_id=-1, matched="")
    for i, ln in lines.iterrows():
        nick, tid = key(ln.player), team(ln.team)
        on_team = players[players.team_id == tid]
        exact = players[players.nick == nick]
        close = difflib.get_close_matches(nick, list(on_team.nick), 1, 0.6)
        # Same team + exact name, then same team + close name, then exact name anywhere (most recent).
        p = (exact[exact.team_id == tid] if (exact.team_id == tid).any() else
             on_team[on_team.nick == close[0]] if close else exact)
        if len(p):
            p = p.iloc[-1]
            out.loc[i, ["player_id", "team_id", "opp_id", "matched"]] = [
                p.player_id, tid if tid != -1 else p.team_id, team(ln.opponent), p.nickname]
    return out


def predict(lines_csv=LINES_CSV, maps_csv=MAPS_CSV):
    df = load(maps_csv)
    lines = pd.read_csv(lines_csv)
    lines = lines[lines.stat.isin(STATS) & (lines.maps.astype(str) == "1-2")].reset_index(drop=True)
    lines["start_date"] = pd.to_datetime(lines.start_time, utc=True, format="ISO8601").fillna(pd.Timestamp.now(tz="UTC"))
    lines = resolve(lines, df).reset_index(names="line_idx")
    known = lines.dropna(subset=["player_id"]).astype({"player_id": int, "team_id": int, "opp_id": int})
    f = features(known[["line_idx", "player_id", "team_id", "opp_id", "start_date"]], df)
    out = lines.merge(f[["line_idx"] + FEATURES], on="line_idx", how="left")
    out["proj"] = out["p_over"] = out["p_under"] = np.nan
    for stat in STATS:
        bundle = joblib.load(f"data/model_{stat}.joblib")
        m = (out.stat == stat) & out[FEATURES].notna().all(axis=1)
        if m.any():
            out.loc[m, "proj"] = bundle["model"].predict(out.loc[m, FEATURES])
            out.loc[m, "p_over"], out.loc[m, "p_under"] = probs(out.loc[m, "proj"], out.loc[m, "line"], bundle["ratios"])
    be_over = out.get("over_odds", pd.Series(np.nan, index=out.index)).fillna(DEFAULT_ODDS).map(american_to_prob)
    be_under = out.get("under_odds", pd.Series(np.nan, index=out.index)).fillna(DEFAULT_ODDS).map(american_to_prob)
    out["edge"] = np.maximum(out.p_over - be_over, out.p_under - be_under)
    out["pick"] = np.select([out.p_over.isna(), out.p_over > be_over, out.p_under > be_under],
                            ["NO DATA", "OVER", "UNDER"], "PASS")
    out = out.sort_values("edge", ascending=False)
    cols = ["book", "player", "matched", "team", "opponent", "stat", "line", "proj", "p_over", "p_under", "edge", "pick"]
    out.to_csv(PICKS_CSV, index=False)
    print(out[[c for c in cols if c in out]].round(3).to_string(index=False))
    print(f"\n{(out.pick == 'NO DATA').sum()} of {len(out)} lines had no usable history (check 'matched').")


if __name__ == "__main__":
    {"train": train, "predict": predict}[sys.argv[1] if len(sys.argv) > 1 else "train"]()
