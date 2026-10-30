"""Features, training and prediction for CS2 "Maps 1-2" kill / headshot props.

    python -m cs2props.model train     # data/maps.csv  -> data/model_{kills,headshots}.joblib
    python -m cs2props.model predict   # data/lines.csv -> data/picks.csv

The tree predicts the stat itself (e.g. maps 1-2 kills). We have no historical prop lines to
label "over/under" with, so P(over line) comes from the tree's projection times the spread of
its held-out errors. Any line, from any book, can then be scored without retraining.
"""
import re
import sys
from itertools import product

import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

MAPS_CSV, LINES_CSV, PICKS_CSV = "data/maps.csv", "data/lines.csv", "data/picks.csv"
STATS = ("kills", "headshots")
FEATURES = ["kills_l20", "kills_l5", "headshots_l20", "hs_pct", "kpr", "adr_l20", "rating_l20",
            "rounds_l20", "maps_played", "map_factor", "win_prob", "elo_opp"]
PRIOR_ROUNDS = 100     # shrink a player's per-map KPR toward their recent KPR by this many rounds
DEFAULT_ODDS = -115    # used when a line has no price (pick'em apps); breakeven 53.5%


def american_to_prob(odds):
    return 100 / (odds + 100) if odds > 0 else -odds / (-odds + 100)


def p_over(pred, line, ratios):
    """P(stat > line) = share of held-out actual/predicted ratios that would clear the line."""
    r = np.sort(ratios)
    return 1 - np.searchsorted(r, np.asarray(line) / np.maximum(pred, 1e-9), side="right") / len(r)


def load(path=MAPS_CSV):
    df = pd.read_csv(path, dtype={"team_id": str, "opp_id": str})
    df["start_date"] = pd.to_datetime(df["start_date"], utc=True, format="ISO8601")
    return df


def build(df):
    """Per player-map features using only maps played *before* that row. Rows must be time-ordered."""
    df = df.sort_values(["start_date", "match_id", "map_number"], kind="stable").reset_index(drop=True)

    # Map-level Elo, frozen at the start of each match (ratings before map 1).
    rating, pre = {}, {}
    for r in df.drop_duplicates("game_id").itertuples():
        a, b = rating.get(r.team_id, 1500.0), rating.get(r.opp_id, 1500.0)
        pre.setdefault((r.match_id, r.team_id), (a, b))
        pre.setdefault((r.match_id, r.opp_id), (b, a))
        if r.win == r.win:  # NaN for upcoming matches
            delta = 32 * (r.win - 1 / (1 + 10 ** ((b - a) / 400)))
            rating[r.team_id], rating[r.opp_id] = a + delta, b - delta
    elo = np.array([pre[k] for k in zip(df.match_id, df.team_id)])
    df["elo_opp"] = elo[:, 1]
    df["win_prob"] = 1 / (1 + 10 ** ((elo[:, 1] - elo[:, 0]) / 400))

    g = df.groupby("player_id")
    for col, n in [("kills", 20), ("kills", 5), ("headshots", 20), ("rounds", 20), ("adr", 20), ("rating", 20)]:
        df[f"{col}_l{n}"] = g[col].transform(lambda s: s.shift().rolling(n, min_periods=min(n, 5)).mean())
    df["kpr"] = df.kills_l20 / df.rounds_l20
    df["hs_pct"] = df.headshots_l20 / df.kills_l20
    df["maps_played"] = g.cumcount()

    # All-time kills/rounds of this player on this map before this row (NaN map name -> NaN).
    keys = [df.player_id, df.map_name]
    k0, r0 = df.kills.fillna(0), df.rounds.fillna(0)
    df["map_kills"] = k0.groupby(keys).cumsum() - k0
    df["map_rounds"] = r0.groupby(keys).cumsum() - r0
    return df


def series_rows(df):
    """One row per (best-of-3+ match, player): features as of map 1, targets summed over maps 1-2."""
    d = df[(df.bo_type >= 3) & df.map_number.isin([1, 2])]
    m1 = d[d.map_number == 1].set_index(["match_id", "player_id"])
    m2 = d[d.map_number == 2].set_index(["match_id", "player_id"])[["kills", "headshots", "map_kills", "map_rounds"]]
    s = m1.join(m2, rsuffix="_2", how="inner")

    def factor(k, r):  # shrunk KPR on this map relative to recent KPR; 1.0 when the map is unknown
        return (((k + PRIOR_ROUNDS * s.kpr) / (r + PRIOR_ROUNDS)) / s.kpr).fillna(1.0)

    s["map_factor"] = (factor(s.map_kills, s.map_rounds) + factor(s.map_kills_2, s.map_rounds_2)) / 2
    for stat in STATS:
        s[f"{stat}_m12"] = s[stat] + s[f"{stat}_2"]
    return s.reset_index()


def fit(X, y, depth, leaf):
    return DecisionTreeRegressor(max_depth=depth, min_samples_leaf=leaf, random_state=0).fit(X, y)


def train(maps_csv=MAPS_CSV):
    s = series_rows(build(load(maps_csv))).dropna(subset=FEATURES).sort_values("start_date")
    n = len(s)
    tr, va, te = s[: int(0.7 * n)], s[int(0.7 * n): int(0.85 * n)], s[int(0.85 * n):]
    print(f"{n} player-series: train {len(tr)} | validate {len(va)} | test {len(te)} "
          f"(test starts {te.start_date.iloc[0]:%Y-%m-%d})")
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
        pred, base = model.predict(te[FEATURES]), 2 * te[f"{stat}_l20"].to_numpy()
        line = np.floor(base) + 0.5  # stand-in line; real books are sharper than this
        po, over = p_over(pred, line, ratios), te[y].to_numpy() > line
        bet_o, bet_u = po > be, po < 1 - be
        bets = bet_o.sum() + bet_u.sum()
        hit = (over[bet_o].sum() + (~over[bet_u]).sum()) / max(bets, 1)
        print(f"{stat}: depth={depth} leaf={leaf} | test MAE tree {np.abs(pred - te[y]).mean():.2f} "
              f"vs 20-map avg {np.abs(base - te[y]).mean():.2f} | stand-in-line bets {bets}/{len(te)} "
              f"hit {hit:.1%} (breakeven {be:.1%}; optimistic, real lines are sharper) | "
              f"Brier {np.mean((po - over) ** 2):.3f} vs 0.250")

        # ponytail: ratios come from the 70%-data model, so production intervals are slightly wide.
        final = fit(pd.concat([tr, va])[FEATURES], pd.concat([tr, va])[y], depth, leaf)
        joblib.dump({"model": final, "ratios": np.sort(ratios)}, f"data/model_{stat}.joblib")
    print("saved data/model_kills.joblib, data/model_headshots.joblib")


def key(x):
    return re.sub(r"[^a-z0-9]", "", str(x).lower())


def resolve(lines, hist):
    """Attach player_id / team_id / opp_id / a fake negative match_id to each line (NaN if unknown)."""
    last = hist.drop_duplicates("player_id", keep="last")
    last = last.assign(nick=last.nickname.map(key), team_key=last.team_name.map(key))
    teams = hist.drop_duplicates("team_id", keep="last")
    team_ids = dict(zip(teams.team_name.map(key), teams.team_id))
    match_ids = {}
    out = lines.assign(player_id=np.nan, team_id=None, opp_id=None, match_id=np.nan)
    for i, ln in lines.iterrows():
        cand = last[last.nick == key(ln.player)]
        if cand.empty:
            continue
        same_team = cand[cand.team_key == key(ln.get("team", ""))]
        p = (same_team if len(same_team) else cand).iloc[-1]  # same team first, then most recent
        opp = team_ids.get(key(ln.get("opponent", "")), f"name:{key(ln.get('opponent', ''))}")
        mid = match_ids.setdefault((frozenset((p.team_id, opp)), ln.start_time), -(len(match_ids) + 1))
        out.loc[i, ["player_id", "team_id", "opp_id", "match_id"]] = [p.player_id, p.team_id, opp, mid]
    return out


def upcoming_rows(resolved):
    """Two unplayed maps per (match, player) so build() computes their pre-match features."""
    rows = []
    for ln in resolved.dropna(subset=["player_id"]).drop_duplicates(["match_id", "player_id"]).itertuples():
        maps = [m.strip() for m in str(ln.maps).split(",")] if pd.notna(getattr(ln, "maps", np.nan)) else []
        for num in (1, 2):
            rows.append(dict(match_id=ln.match_id, game_id=-(len(rows) + 1), start_date=ln.start_time, bo_type=3,
                             map_number=num, map_name=maps[num - 1] if len(maps) >= num else np.nan,
                             team_id=ln.team_id, opp_id=ln.opp_id, player_id=ln.player_id))
    return pd.DataFrame(rows)


def predict(lines_csv=LINES_CSV, maps_csv=MAPS_CSV):
    lines = pd.read_csv(lines_csv)
    lines["start_time"] = pd.to_datetime(lines.start_time, utc=True, format="ISO8601")
    hist = load(maps_csv)
    lines = resolve(lines, hist)
    if lines.player_id.isna().all():
        sys.exit("no line matched a player in data/maps.csv")
    up = upcoming_rows(lines)
    feats = series_rows(build(pd.concat([hist, up], ignore_index=True)))
    feats = feats[feats.match_id < 0][["match_id", "player_id"] + FEATURES]
    out = lines.merge(feats, on=["match_id", "player_id"], how="left")
    out["proj"] = out["p_over"] = np.nan
    for stat in STATS:
        bundle = joblib.load(f"data/model_{stat}.joblib")
        m = (out.stat == stat) & out[FEATURES].notna().all(axis=1)
        if m.any():
            out.loc[m, "proj"] = bundle["model"].predict(out.loc[m, FEATURES])
            out.loc[m, "p_over"] = p_over(out.loc[m, "proj"].to_numpy(), out.loc[m, "line"].to_numpy(),
                                          bundle["ratios"])
    over_be = out.get("over_odds", pd.Series(DEFAULT_ODDS, index=out.index)).fillna(DEFAULT_ODDS).map(american_to_prob)
    under_be = out.get("under_odds", pd.Series(DEFAULT_ODDS, index=out.index)).fillna(DEFAULT_ODDS).map(american_to_prob)
    out["edge"] = np.maximum(out.p_over - over_be, (1 - out.p_over) - under_be)
    out["pick"] = np.select([out.p_over.isna(), out.p_over > over_be, 1 - out.p_over > under_be],
                            ["NO DATA", "OVER", "UNDER"], "PASS")
    out = out.sort_values("edge", ascending=False)
    cols = [c for c in ["player", "team", "opponent", "book", "stat", "line", "proj", "p_over", "edge", "pick"]
            if c in out]
    out.to_csv(PICKS_CSV, index=False)
    print(out[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    {"train": train, "predict": predict}[sys.argv[1] if len(sys.argv) > 1 else "train"]()
