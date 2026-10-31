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
    model.train()
    when = "2026-01-01T00:00:00Z"
    pd.DataFrame([  # the book calls player 'max' "MaxOff" and the team "Team Alpha Esports"
        dict(book="a", player="MaxOff", team="Team Alpha Esports", opponent="Delta", stat="kills", maps="1-2",
             line=30.5, start_time=when),
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
    assert ("p7", "kills") not in picks.index                        # maps 1-3 is not modelled


def test_backtest_on_past_lines(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    df = fake_maps(400)
    df.to_csv("data/maps.csv", index=False)
    s = model.series(df)
    late = s[(s.match_id >= 300) & s.player_id.isin([0, 7])]  # 'max' averages ~40, 'p7' ~30
    names = dict(enumerate(TEAMS))

    def run(shift):  # the book lists the match `shift` away from bo3.gg's start time
        pd.DataFrame(dict(book="pf", player=late.player_id.map({0: "max", 7: "p7"}), team=late.team_id.map(names),
                          opponent=late.opp_id.map(names), stat="kills", maps="1-2",
                          line=late.player_id.map({0: 30.5, 7: 36.5}), over_odds=-115, under_odds=-115,
                          start_time=(late.start_date + shift).dt.strftime("%Y-%m-%dT%H:%M:%SZ"), actual=late.kills_m12)
                     ).to_csv("data/prop_history.csv", index=False)
        model.backtest()
        return pd.read_csv("data/backtest.csv").sort_values(["player", "match_start"]).reset_index(drop=True)

    early, bt = run(pd.Timedelta(minutes=-5)), run(pd.Timedelta(minutes=5))
    assert "agrees with bo3.gg on 100.0%" in capsys.readouterr().out
    pd.testing.assert_series_equal(early.proj, bt.proj)  # listing the match late must not leak its result
    assert len(bt) >= 0.95 * len(late)  # a few series fail load()'s sanity checks and are not graded
    assert (bt[bt.player == "max"].pick == "OVER").mean() > 0.9
    assert (bt[bt.player == "p7"].pick == "UNDER").mean() > 0.9
    assert bt.units.sum() > 0


def test_pickfinder_history_rows():
    from cs2props import pickfinder

    def game(start, vs, abbr, kills, score=(2, 0), **extra):
        return dict(startTime=start, team="Nemiga", vs=vs, vsAbbr=abbr, scores=[dict(score=score[0]), dict(score=score[1])],
                    periodStats=[dict(kills=k, headshots=k // 2) for k in kills], **extra)

    def prop(start, opp, abbr, line=27.5, outcomes=None):
        return dict(stat="Maps 1-2 Kills", line=line, start_date=start, team=dict(name="Nemiga"),
                    opponent=dict(name=opp, abbreviation=abbr),
                    apps=dict(prizepicks=dict(outcomes=outcomes or dict(over=dict(price=-115), under=dict(price=-110)))))

    games = [game("2025-10-30 12:02:35.000Z", "Vitality", "VIT", [17, 13], lines={"Maps 1-2 Kills": 27.5}),
             game("2025-09-07 11:07:00.000Z", "Copenhagen Wolves", "CW", [15, 15]),   # same-day rematch ...
             game("2025-09-07 08:06:00.000Z", "Copenhagen Wolves", "CW", [20, 17]),   # ... the one at 08:00
             game("2025-09-01 12:00:00.000Z", "Sat out", "SO", [20, 20], score=(2, 1)),  # missed a map: not maps 1-2
             game("2025-10-28 12:00:00.000Z", "Bo1 side", "X", [20])]
    props = [prop("2025-10-30T12:00:00Z", "Team Vitality", "VITALITY"),        # abbreviations differ: match by name
             prop("2025-09-07T08:00:00Z", "Copenhagen Wolves", "CW"),
             prop("2025-10-31T11:00:00Z", "FURIA", "FURIA"),                    # result not logged yet: no row
             prop("2025-09-01T12:00:00Z", "Sat out", "SO"),
             prop("2025-10-30T12:00:00Z", "Vitality", "VIT", line=33.5,         # demon: over only, can't bet under
                  outcomes=dict(over=dict(price=-110, modifier="demon")))]
    rows = pickfinder.player_rows(dict(player_name="KaiR0N", team=dict(name="New team")), games, props,
                                  pd.Timestamp("2025-05-01", tz="UTC"))
    assert [(r["source"], r["opponent"], r["line"], r["actual"], r["team"]) for r in rows] == [
        ("gamelog", "Vitality", 27.5, 30, "Nemiga"),
        ("props", "Team Vitality", 27.5, 30, "Nemiga"),
        ("props", "Copenhagen Wolves", 27.5, 37, "Nemiga")]
    assert (rows[1]["book"], rows[1]["over_odds"], rows[1]["under_odds"]) == ("prizepicks", -115, -110)
