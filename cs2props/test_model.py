"""python -m pytest cs2props -q"""
import numpy as np
import pandas as pd

from cs2props import lines, model

TEAMS = ["Alpha Gaming", "Bravo", "Charlie", "Delta"]


def fake_maps(n_matches=400, seed=0):
    """BO3s between 4 teams of 5. Alpha (team 0) is much stronger; player 0 'maxoff' gets more kills."""
    rng = np.random.default_rng(seed)
    rows, game = [], 0
    for m in range(n_matches):
        t1, t2 = (int(t) for t in rng.choice(4, 2, replace=False))
        for num, mp in enumerate(rng.choice(["de_nuke", "de_ancient", "de_inferno", "de_mirage"], 3, replace=False), 1):
            game += 1
            w1 = rng.random() < (0.85 if t1 == 0 else 0.15 if t2 == 0 else 0.5)
            for t, o, won in ((t1, t2, w1), (t2, t1, not w1)):
                for k in range(5):
                    pid = t * 5 + k
                    kills = int(rng.poisson(20 if pid == 0 else 15))
                    rows.append(dict(match_id=m, start_date=pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(hours=m),
                                     tier="b", bo_type=3, game_id=game, map_number=num, map_name=mp, rounds=22,
                                     team_id=t, team_name=TEAMS[t], opp_id=o, player_id=pid,
                                     nickname="max" if pid == 0 else f"p{pid}", win=int(won), kills=kills,
                                     deaths=15, headshots=kills // 2, assists=4, adr=80.0, kast=70))
    return pd.DataFrame(rows)


def test_features_never_see_the_match_being_predicted():
    df = fake_maps(60)
    a = model.features(model.series(df), df)
    df.loc[df.match_id == 59, ["kills", "headshots"]] += 100
    b = model.features(model.series(df), df)
    cols = ["match_id", "player_id"] + model.FEATURES
    pd.testing.assert_frame_equal(a[a.match_id == 59][cols], b[b.match_id == 59][cols])
    assert (b[b.match_id == 59].kills_m12.to_numpy() > a[a.match_id == 59].kills_m12.to_numpy()).all()


def test_elo_probabilities_and_parsing():
    df = fake_maps(400)
    s = model.features(model.series(df), df)
    assert s[(s.match_id > 300) & (s.team_id == 0)].win_prob.mean() > 0.7
    later = df.match_id >= 300  # Alpha's five move to a new org (team id 9): the rating should follow them
    moved = df.assign(team_id=df.team_id.mask(later & (df.team_id == 0), 9), opp_id=df.opp_id.mask(later & (df.opp_id == 0), 9))
    s9 = model.features(model.series(moved), moved).query("team_id == 9")
    assert s9.win_prob.iloc[0] > 0.7
    over, under = model.probs([30.0, 30.0, 30.0], [30.5, 30.5, 30.0], np.array([-1.0, 0.0, 1.0]))
    assert np.allclose(over, [1 / 3] * 3) and np.allclose(under, [2 / 3, 2 / 3, 1 / 3])  # 30.0 line: push 1/3
    assert round(model.american_to_prob(-115), 4) == 0.5349
    rounds = model.map_rounds(np.array([0.5, 0.7, 0.9]))
    assert rounds[0] > rounds[1] > rounds[2] > 20  # close maps run longer
    assert np.allclose(model.series_to_map(pd.Series([0.5, 0.6 ** 2 * 1.8, np.nan])), [0.5, 0.6, np.nan], equal_nan=True)

    class Half:  # a calibrator that says every line is a coin flip
        def predict_proba(self, X):
            return np.full((len(X), 2), 0.5)

    two = pd.DataFrame(dict(proj=[30.0, 30.0], line=[30.0, 31.5], stat="kills", p_over=[0.4, 0.4],
                            p_under=[0.5, 0.6], over_odds=-110, under_odds=-110))
    out = model.decide(two, Half())
    assert np.allclose(out.p_over, [0.45, 0.5]) and np.allclose(out.p_under, [0.45, 0.5])  # 10% push kept
    assert (out.pick == "PASS").all()
    # Pushes are refunded: 53% over / 39% under on a whole-number line is 57.6% when it is decided, so OVER at -119.
    whole = pd.DataFrame(dict(p_over=[0.53], p_under=[0.39], over_odds=-119, under_odds=-119))
    assert model.decide(whole).pick.iloc[0] == "OVER"
    assert lines.market("MAPS 1-2 Kills") == lines.market("kills_on_maps_1_2") == {"stat": "kills", "maps": "1-2"}
    assert lines.market("1st_3_maps_player_headshots") == {"stat": "headshots", "maps": "1-3"}
    assert lines.market("2nd_map_player_kills") == {"stat": "kills", "maps": "2"}
    assert lines.american(1.78) == -128 and lines.american(2.5) == 150


def test_load_drops_incomplete_box_scores(tmp_path):
    df = fake_maps(3)
    df.loc[df.game_id == 1, "headshots"] = np.nan  # partial live snapshot
    df.loc[df.game_id == 2, "headshots"] = 999     # glitch: more headshots than kills
    df.loc[df.game_id == 3, "rounds"] = 5          # forfeit
    df.to_csv(tmp_path / "maps.csv", index=False)
    kept = model.load(tmp_path / "maps.csv")
    assert set(kept.game_id) == set(df.game_id) - {1, 2, 3}
    assert model.series(kept).match_id.tolist().count(0) == 0  # match 0 lost maps 1-2, so no label


def test_train_then_predict(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    fake_maps(400).to_csv("data/maps.csv", index=False)
    model.compare()  # both model kinds still fit on the same split
    model.train()
    when = "2026-01-01T00:00:00Z"
    pd.DataFrame([  # the book calls player 'max' "MaxOff" and the team "Team Alpha Esports"
        dict(book="a", player="MaxOff", team="Team Alpha Esports", opponent="Delta", stat="kills", maps="1-2",
             line=30.5, start_time=when),
        dict(book="pickfinder", player="max", team="Alpha", opponent="Delta", stat="match_odds", maps="",
             start_time=when, win_market=0.8),  # odds-only row: every line of this match shares it
        dict(book="b", player="maxoff", team="Alpha", opponent="Delta", stat="headshots", maps="1-2",
             line=25.5, over_odds=-112, under_odds=-112, start_time=when),
        dict(book="a", player="p7", team="Bravo", opponent="Charlie", stat="kills", maps="1-3", line=45.5, start_time=when),
        dict(book="a", player="nobody", team="X", opponent="Y", stat="kills", maps="1-2", line=30.5, start_time=when),
        # exact name on another team beats a close name on this team ("p10" is on Charlie)
        dict(book="a", player="p1", team="Charlie", opponent="Delta", stat="kills", maps="1-2", line=20.5, start_time=when),
    ]).to_csv("data/lines.csv", index=False)
    model.predict()
    picks = pd.read_csv("data/picks.csv").set_index(["player", "stat"])
    assert picks.loc[("MaxOff", "kills"), "matched"] == "max"
    assert picks.loc[("MaxOff", "kills"), "pick"] == "OVER"         # ~40 expected vs 30.5
    assert picks.loc[("maxoff", "headshots"), "pick"] == "UNDER"    # ~20 expected vs 25.5
    assert picks.loc[("nobody", "kills"), "pick"] == "NO DATA"
    assert picks.loc[("p1", "kills"), "matched"] == "p1 (Alpha Gaming)"
    p_map = model.series_to_map(pd.Series([0.8]))[0]
    assert np.isclose(picks.loc[("MaxOff", "kills"), "win_prob"], p_map)
    assert np.isclose(picks.loc[("maxoff", "headshots"), "win_prob"], p_map)  # same match, no odds of its own
    assert ("p7", "kills") not in picks.index                        # maps 1-3 is not modelled


def test_backtest_on_past_lines(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    df = fake_maps(400)
    df.to_csv("data/maps.csv", index=False)
    s = model.series(df)
    late = s[s.match_id >= 250]  # 'max' (player 0) averages ~40 kills over maps 1-2, everyone else ~30
    names = dict(enumerate(TEAMS))

    def run(shift):  # the book lists the match `shift` away from bo3.gg's start time
        pd.DataFrame(dict(book="pf", player=["max" if p == 0 else f"p{p}" for p in late.player_id],
                          team=late.team_id.map(names), opponent=late.opp_id.map(names), stat="kills", maps="1-2",
                          line=np.where(late.player_id == 0, 30.5, 36.5), over_odds=-115, under_odds=-115,
                          start_time=(late.start_date + shift).dt.strftime("%Y-%m-%dT%H:%M:%SZ"), actual=late.kills_m12)
                     ).to_csv("data/prop_history.csv", index=False)
        model.backtest()
        return pd.read_csv("data/backtest.csv").sort_values(["player", "match_start"]).reset_index(drop=True)

    early, bt = run(pd.Timedelta(minutes=-5)), run(pd.Timedelta(minutes=5))
    printed = capsys.readouterr().out
    assert "agrees with bo3.gg on 100.0%" in printed and "log loss" in printed
    saved = "Saved data/calibrator.joblib" in printed
    assert saved or "did not beat the raw model" in printed  # saved only when it wins on later matches
    pd.testing.assert_series_equal(early.proj, bt.proj)  # listing the match late must not leak its result
    assert len(bt) >= 0.95 * len(late)  # a few series fail load()'s sanity checks and are not graded
    assert (bt[bt.player == "max"].pick == "OVER").mean() > 0.9
    assert (bt[bt.player != "max"].pick == "UNDER").mean() > 0.9
    assert bt.units.sum() > 0
    model.train()
    pd.DataFrame([dict(book="a", player="max", team="Alpha Gaming", opponent="Delta", stat="kills", maps="1-2",
                       line=30.5, start_time="2026-01-01T00:00:00Z")]).to_csv("data/lines.csv", index=False)
    model.predict()
    assert ("calibrated on" in capsys.readouterr().out) == saved


def test_pickfinder_history_rows():
    from cs2props import pickfinder

    def game(start, vs, vs_id, kills, score=(2, 0)):
        return dict(startTime=start, team="G2", vs=vs, vsId=vs_id, scores=[dict(score=score[0]), dict(score=score[1])],
                    periodStats=[dict(kills=k, headshots=k // 2) for k in kills])

    def side(price, link=None, modifier=""):
        return dict(price=price, modifier=modifier, deep_link=link and dict(desktop=link))

    def entry(key, line, **apps):
        return dict(matchKey=key, stat="Maps 1-2 Kills", line=line, apps=apps)

    pp = "https://app.prizepicks.com/?projections=16194603-{}-28.5"
    games = [game("2025-10-26 16:34:00.000Z", "DENDELE CS", "138988", [16, 14, 20], score=(2, 1)),  # played all 3
             game("2025-11-02 05:31:03.000Z", "PARIVISION", "133719", [12, 8]),        # scheduled 11/01 ET
             game("2025-09-07 08:06:00.000Z", "Copenhagen Wolves", "500", [20, 17]),   # two series vs the same
             game("2025-09-07 11:07:00.000Z", "Copenhagen Wolves", "500", [15, 15]),   # ... team that day
             game("2025-09-01 12:00:00.000Z", "Sat out", "77", [20, 20], score=(2, 1))]  # missed a map
    hist = {"Maps 1-2 Kills": [
        dict(entry("2025-10-26-138988", 26.5,  # PrizePicks' own line (28.5) is in its link; Underdog's the entry's
              prizepicks=dict(outcomes=dict(over=side(-119, pp.format("o")), under=side(-119, pp.format("u")))),
              underdog=dict(outcomes=dict(over=side(-115), under=side(-115))),
              goblin=dict(outcomes=dict(over=side(-155, modifier="goblin")))),
             teamOdds=dict(implied=0.73, american=-385), opponentOdds=dict(implied=0.27)),
        entry("2025-11-01-133719", 26.5, underdog=dict(outcomes=dict(over=side(-115), under=side(-110)))),
        entry("2025-11-09-133719", 25.5, underdog=dict(outcomes=dict(over=side(-115), under=side(-115)))),  # not played
        entry("2025-09-07-500", 27.5, underdog=dict(outcomes=dict(over=side(-115), under=side(-115)))),  # ambiguous
        entry("2025-09-01-77", 27.5, underdog=dict(outcomes=dict(over=side(-115), under=side(-115)))),
        entry("2025-01-05-138988", 27.5, underdog=dict(outcomes=dict(over=side(-115), under=side(-115))))]}  # too old
    rows = pickfinder.player_rows(dict(player_name="huNter-"), games, hist, pd.Timestamp("2025-05-01").date())
    assert [(r["book"], r["opponent"], r["line"], r["actual"], r["under_odds"]) for r in rows] == [
        ("prizepicks", "DENDELE CS", 28.5, 30, -119), ("underdog", "DENDELE CS", 26.5, 30, -115),
        ("underdog", "PARIVISION", 26.5, 20, -110)]
    assert rows[0]["start_time"].startswith("2025-10-26T16:34") and rows[0]["team"] == "G2"
    assert rows[0]["win_market"] == 0.73 and rows[2]["win_market"] is None


def test_lines_archive_survives_a_new_column(tmp_path):
    path = tmp_path / "lines_history.csv"
    pd.DataFrame(dict(book=["a"], line=[30.5], scraped_at=["t0"])).to_csv(path, index=False)  # older version
    lines.append_csv(pd.DataFrame(dict(book=["b"], line=[25.5], win_market=[0.6], scraped_at=["t1"])), path)
    lines.append_csv(pd.DataFrame(dict(book=["c"], line=[27.5], win_market=[0.4], scraped_at=["t2"])), path)
    got = pd.read_csv(path)
    assert got.columns.tolist() == ["book", "line", "win_market", "scraped_at"]
    assert got.book.tolist() == ["a", "b", "c"] and got.win_market.isna().tolist() == [True, False, False]


def test_calibrator_saved_only_when_it_beats_the_raw_model(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    df = fake_maps(400)
    df.to_csv("data/maps.csv", index=False)
    s = model.series(df)
    late = s[s.match_id >= 250]
    names = dict(enumerate(TEAMS))
    pd.DataFrame(dict(book="pf", player=["max" if p == 0 else f"p{p}" for p in late.player_id],
                      team=late.team_id.map(names), opponent=late.opp_id.map(names), stat="kills", maps="1-2",
                      line=np.where(late.player_id == 0, 39.5, 29.5), over_odds=-115, under_odds=-115,
                      start_time=late.start_date.dt.strftime("%Y-%m-%dT%H:%M:%SZ"))).to_csv("data/props.csv", index=False)
    # A raw model that is sure every line goes over (lines sit at the players' means, so ~half do): the
    # calibrator learned from real outcomes must beat it, get saved, and be used by predict.
    monkeypatch.setattr(model, "probs", lambda pred, line, z: (np.full(len(pred), 0.95), np.full(len(pred), 0.05)))
    model.backtest("data/props.csv")
    assert "Saved data/calibrator.joblib" in capsys.readouterr().out
    model.train()
    pd.DataFrame([dict(book="a", player="max", team="Alpha Gaming", opponent="Delta", stat="kills", maps="1-2",
                       line=39.5, start_time="2026-01-01T00:00:00Z")]).to_csv("data/lines.csv", index=False)
    model.predict()
    printed = capsys.readouterr().out
    assert "calibrated on" in printed and "data/props.csv" in printed
    p = pd.read_csv("data/picks.csv").p_over.iloc[0]
    assert 0.3 < p < 0.7  # the calibrator pulled the raw 0.95 back toward what actually happened
