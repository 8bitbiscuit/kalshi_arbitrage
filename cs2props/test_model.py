"""python -m pytest cs2props -q"""
import joblib
import numpy as np
import pandas as pd

from cs2props import lines, model

TEAMS = ["Alpha Gaming", "Bravo", "Charlie", "Delta"]


def fake_maps(n_matches=400, seed=0, hours=1):
    """BO3s between 4 teams of 5, `hours` apart. Alpha (team 0) is much stronger; player 0 'maxoff' gets more kills."""
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
                    rows.append(dict(match_id=m, start_date=pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(hours=m * hours),
                                     tier="b", bo_type=3, game_id=game, map_number=num, map_name=mp, rounds=22,
                                     team_id=t, team_name=TEAMS[t], opp_id=o, player_id=pid,
                                     nickname="max" if pid == 0 else f"p{pid}", win=int(won), kills=kills,
                                     deaths=15, headshots=kills // 2, assists=4, adr=80.0, kast=70))
    return pd.DataFrame(rows)


def test_features_never_see_the_match_being_predicted():
    df = fake_maps(60)
    a = model.features(model.series(df), df)
    df.loc[df.match_id == 59, ["kills", "headshots"]] += 100
    df.loc[df.match_id == 59, "win"] = 1 - df.win  # results move Elo, which sched_gap uses
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
             line=30.5, start_time=when, market_win_prob=0.3),
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
    assert picks.loc[("MaxOff", "kills"), "win_prob"] == 0.3 and picks.loc[("MaxOff", "kills"), "elo_win_prob"] > 0.7
    assert picks.loc[("MaxOff", "kills"), "pick"] == "OVER"         # ~40 expected vs 30.5
    assert picks.loc[("maxoff", "headshots"), "pick"] == "UNDER"    # ~20 expected vs 25.5
    assert picks.loc[("nobody", "kills"), "pick"] == "NO DATA"
    assert picks.loc[("p1", "kills"), "matched"] == "p1 (Alpha Gaming)"
    assert ("p7", "kills") not in picks.index                        # maps 1-3 is not modelled


def test_backtest_on_past_lines(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(model, "MIN_CAL_LINES", 20)
    (tmp_path / "data").mkdir()
    df = fake_maps(400, hours=6)  # graded lines span March and April: two monthly refits, April calibrated on March
    df.to_csv("data/maps.csv", index=False)
    s = model.series(df)
    late = s[(s.match_id >= 300) & s.player_id.isin([0, 7])]  # 'max' averages ~40, 'p7' ~30
    names = dict(enumerate(TEAMS))

    def run(shift):  # the book lists the match `shift` away from bo3.gg's start time
        pf = pd.DataFrame(dict(book="pf", player=late.player_id.map({0: "max", 7: "p7"}), team=late.team_id.map(names),
                               opponent=late.opp_id.map(names), stat="kills", maps="1-2",
                               line=late.player_id.map({0: 30.5, 7: 36.5}), over_odds=-115, under_odds=-115,
                               start_time=(late.start_date + shift).dt.strftime("%Y-%m-%dT%H:%M:%SZ"), actual=late.kills_m12))
        pp = pf[pf.player == "max"].assign(book="pp", line=31.5)  # a second book a kill higher on max
        pd.concat([pf, pp]).to_csv("data/prop_history.csv", index=False)
        model.backtest()
        return pd.read_csv("data/backtest.csv").sort_values(["player", "match_start"]).reset_index(drop=True)

    early, bt = run(pd.Timedelta(minutes=-5)), run(pd.Timedelta(minutes=5))
    out = capsys.readouterr().out
    assert "agrees with bo3.gg on 100.0%" in out and "Calibrated on earlier months' lines" in out
    assert "Line shopping (untimed lines)" in out and "disagree by 1.0 on average" in out
    pd.testing.assert_series_equal(early.proj, bt.proj)  # listing the match late must not leak its result
    assert len(bt) >= 0.95 * len(late)  # a few series fail load()'s sanity checks and are not graded
    assert (bt[bt.player == "max"].pick == "OVER").mean() > 0.9
    assert (bt[bt.player == "p7"].pick == "UNDER").mean() > 0.9
    assert bt.units.sum() > 0
    april = bt.month == "2025-04"
    assert bt.pick_cal[~april].isna().all() and (bt.pick_cal[april] == bt.pick[april]).mean() > 0.9
    saved = joblib.load(model.CAL_JOBLIB)
    assert saved["kind"] == "gbm" and saved["lines"]["kills"] > 100 and set(saved["cal"]) == {"kills"}
    monkeypatch.setattr(model, "MIN_CAL_LINES", 10_000)  # too few lines now: the saved calibration must survive
    run(pd.Timedelta(minutes=5))
    assert "Calibration not saved" in capsys.readouterr().out and joblib.load(model.CAL_JOBLIB)["lines"] == saved["lines"]


def test_calibration_shrinks_an_overconfident_model():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.2, 0.8, 6000)  # the model's P(over); the truth leans over and moves 40% as much
    over = rng.random(6000) < 0.54 + 0.4 * (p - 0.5)
    g = pd.DataFrame(dict(p_over=p, p_under=1 - p, stat="kills", result=np.where(over, "OVER", "UNDER")))
    cal = model.fit_calibration(g)
    assert set(cal) == {"kills"} and 0.25 < cal["kills"].coef_[0, 0] < 0.6  # slope in logit units: ~0.4
    c = model.calibrated(g.copy(), cal)
    assert abs(c.p_over.mean() - over.mean()) < 0.01 and c.p_over.std() < 0.5 * p.std()


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
        entry("2025-10-26-138988", 26.5,  # PrizePicks' own line (28.5) is in its link; Underdog's is the entry's
              prizepicks=dict(outcomes=dict(over=side(-119, pp.format("o")), under=side(-119, pp.format("u")))),
              underdog=dict(outcomes=dict(over=side(-115), under=side(-115))),
              goblin=dict(outcomes=dict(over=side(-155, modifier="goblin")))),
        {**entry("2025-11-01-133719", 26.5, underdog=dict(outcomes=dict(over=side(-115), under=side(-110)))),
         "teamOdds": dict(implied=0.6154, american=-200), "opponentOdds": dict(implied=0.3846)},
        entry("2025-11-09-133719", 25.5, underdog=dict(outcomes=dict(over=side(-115), under=side(-115)))),  # not played
        entry("2025-09-07-500", 27.5, underdog=dict(outcomes=dict(over=side(-115), under=side(-115)))),  # ambiguous
        entry("2025-09-01-77", 27.5, underdog=dict(outcomes=dict(over=side(-115), under=side(-115)))),
        entry("2025-01-05-138988", 27.5, underdog=dict(outcomes=dict(over=side(-115), under=side(-115))))]}  # too old
    rows = pickfinder.player_rows(dict(player_name="huNter-"), games, hist, pd.Timestamp("2025-05-01").date())
    assert [(r["book"], r["opponent"], r["line"], r["actual"], r["under_odds"]) for r in rows] == [
        ("prizepicks", "DENDELE CS", 28.5, 30, -119), ("underdog", "DENDELE CS", 26.5, 30, -115),
        ("underdog", "PARIVISION", 26.5, 20, -110)]
    assert rows[0]["start_time"].startswith("2025-10-26T16:34") and rows[0]["team"] == "G2"
    assert np.isnan(rows[0]["market_win_prob"]) and abs(rows[2]["market_win_prob"] - 0.5776) < 0.001


def test_market_moneyline():
    from cs2props import pickfinder
    # A real props item (Legacy vs B8, 2026-11-03): team_ml -200, opponent_ml +140, team_ml_percent 0.6154
    legacy, b8 = pickfinder.map_win_prob(0.6154), pickfinder.map_win_prob(0.3846)
    assert abs(legacy - 0.5776) < 0.001 and abs(legacy + b8 - 1) < 1e-3 and pickfinder.map_win_prob(0.5) == 0.5
    assert all(np.isnan(pickfinder.map_win_prob(x)) for x in (None, 0, 1, float("nan")))  # missing, not a certainty
    # BetBoom vs FURIA, 2026-11-06: the props item, its bare moneylines, and the Prop History entry all agree
    props = dict(team_ml=138, opponent_ml=-196, team_ml_percent=0.3882, opponent_ml_percent=0.6118)
    assert pickfinder.series_win_prob(props) == 0.3882
    assert abs(pickfinder.series_win_prob(dict(team_ml=138, opponent_ml=-196)) - 0.3882) < 1e-4
    assert pickfinder.series_win_prob(dict(teamOdds=dict(implied=0.3882, american=138))) == 0.3882
    assert np.isnan(pickfinder.series_win_prob(dict(matchKey="2026-11-04-3213", line=14.5)))  # entry without odds


def test_pickfinder_lines_pull_the_moneyline(monkeypatch):
    from contextlib import contextmanager
    from cs2props import pickfinder
    item = dict(fixture_id="2026110686A63993", player_id="bn3mlvy0cst11z3", player_name="s1ren", isAltLine=False,
                team=dict(name="BetBoom Team"), opponent=dict(name="FURIA"), line=26.5, start_date="2026-11-06T09:45:00Z",
                market_id="1st_2_maps_player_kills",
                apps=dict(dabble=dict(outcomes=dict(over=dict(price=-122, modifier=""), under=dict(price=-122, modifier="")))))
    calls = []

    @contextmanager
    def session(priced):  # /props without the moneyline unless `priced`; a player's own props always carry it
        def get(path, **params):
            calls.append((path, params.get("moneylines")))
            ml = dict(team_ml=138, opponent_ml=-196, team_ml_percent=0.3882)
            return {"items": [{**item, **(ml if priced else {})}], "totalPages": 1} if path == "/props" else [{**item, **ml}]
        yield get

    for priced in (False, True):
        calls.clear()
        lines.MARKET.clear()
        monkeypatch.setattr(pickfinder, "session", lambda: session(priced))
        rows = list(lines.pickfinder())
        assert [r["book"] for r in rows] == ["dabble"] and abs(lines.MARKET["betboom|furia"] - 0.4249) < 1e-3
        assert calls == [("/props", 1)] + ([] if priced else [("/players/bn3mlvy0cst11z3/props", None)])


def test_lines_market_and_history(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    # an archive written before market_win_prob existed must keep its rows
    pd.DataFrame([dict(book="sleeper", player="old", team="X", opponent="Y", stat="kills", maps="1-2", line=20.5,
                       over_odds=-110, under_odds=-110, start_time="2026-10-01T10:00:00Z",
                       scraped_at="2026-10-01T09:00:00Z")]).to_csv(lines.HISTORY_CSV, index=False)
    row = dict(stat="kills", maps="1-2", line=30.5, start_time="2026-11-03T02:00:00Z")
    monkeypatch.setattr(lines, "prizepicks", lambda: [dict(book="prizepicks", player="huNter-", team="G2",
                                                           opponent="Aurora", **row)])
    monkeypatch.setattr(lines, "underdog", lambda: [dict(book="underdog", player="woxic", team="Aurora Gaming",
                                                         opponent="G2 Esports", **row)])
    monkeypatch.setattr(lines, "sleeper", lambda: [])
    monkeypatch.setattr(lines, "pickfinder", lambda: lines.MARKET.update({"g2|aurora": 0.45}) or [])
    monkeypatch.setattr(lines.sys, "argv", ["lines"])
    monkeypatch.setenv("PICKFINDER_EMAIL", "me@example.com")  # a login set: PickFinder runs by default
    lines.main()
    hist = pd.read_csv(lines.HISTORY_CSV)
    assert list(hist.player) == ["old", "huNter-", "woxic"] and hist.market_win_prob.isna().tolist() == [True, False, False]
    assert pd.read_csv(lines.LINES_CSV).market_win_prob.round(2).tolist() == [0.45, 0.55]  # opponent's side: 1 - p
