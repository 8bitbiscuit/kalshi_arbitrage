"""python -m pytest cs2props -q"""
import numpy as np
import pandas as pd

from cs2props import model


def fake_maps(n_matches=400, seed=0):
    """BO3s between 4 teams of 5; team 'A' is much stronger, 'koala' (pid 0) gets more kills."""
    rng = np.random.default_rng(seed)
    teams, rows, game = list("ABCD"), [], 0
    for m in range(n_matches):
        t1, t2 = rng.choice(4, 2, replace=False)
        for num, mp in enumerate(rng.choice(["de_nuke", "de_ancient", "de_inferno", "de_mirage"], 3, replace=False), 1):
            game += 1
            p1 = 0.85 if t1 == 0 else 0.15 if t2 == 0 else 0.5
            w1 = rng.random() < p1
            for t, o, won in ((t1, t2, w1), (t2, t1, not w1)):
                for k in range(5):
                    pid = t * 5 + k
                    kills = rng.poisson(20 if pid == 0 else 15)
                    rows.append(dict(match_id=m, start_date=pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(hours=m),
                                     bo_type=3, game_id=game, map_number=num, map_name=mp, rounds=22,
                                     team_id=teams[t], team_name=f"Team {teams[t]}", opp_id=teams[o],
                                     opp_name=f"Team {teams[o]}", player_id=pid, nickname="koala" if pid == 0 else f"p{pid}",
                                     win=int(won), kills=kills, headshots=kills // 2, deaths=15, adr=80.0, rating=1.0))
    return pd.DataFrame(rows)


def test_features_do_not_peek_at_the_match_being_predicted():
    df = fake_maps(60)
    a = model.series_rows(model.build(df))
    df.loc[df.match_id == 59, "kills"] += 100
    b = model.series_rows(model.build(df))
    last_a, last_b = a[a.match_id == 59], b[b.match_id == 59]
    pd.testing.assert_frame_equal(last_a[model.FEATURES], last_b[model.FEATURES])
    assert (last_b.kills_m12 > last_a.kills_m12).all()


def test_elo_and_probability_math():
    s = model.series_rows(model.build(fake_maps(400)))
    late = s[s.match_id > 300]
    assert late[late.team_id == "A"].win_prob.mean() > 0.7
    assert model.p_over(np.array([30.0]), [30.5], np.array([1.0]))[0] == 0
    assert model.p_over(np.array([30.0]), [30.5], np.array([0.9, 1.1]))[0] == 0.5
    assert round(model.american_to_prob(-115), 4) == 0.5349


def test_train_then_predict(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    fake_maps(400).to_csv("data/maps.csv", index=False)
    model.train()
    pd.DataFrame([
        dict(player="Koala", team="Team A", opponent="Team D", stat="kills", line=30.5, start_time="2026-01-01T00:00Z",
             book="test", maps="de_nuke,de_ancient"),
        dict(player="koala", team="Team A", opponent="Team D", stat="headshots", line=25.5,
             start_time="2026-01-01T00:00Z", book="test"),
        dict(player="nobody", team="X", opponent="Y", stat="kills", line=30.5, start_time="2026-01-01T00:00Z"),
    ]).to_csv("data/lines.csv", index=False)
    model.predict()
    picks = pd.read_csv("data/picks.csv").set_index("stat", append=True)
    kills = picks.xs("kills", level="stat")
    assert kills.set_index("player").loc["Koala", "pick"] == "OVER"   # ~40 expected vs 30.5
    assert kills.set_index("player").loc["nobody", "pick"] == "NO DATA"
    assert picks.xs("headshots", level="stat").iloc[0]["pick"] == "UNDER"  # ~20 expected vs 25.5
