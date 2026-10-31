"""Fetch current CS2 player-prop lines into data/lines.csv and append them to data/lines_history.csv.

    python -m cs2props.lines               # PrizePicks + Underdog + Sleeper public JSON feeds, no login
    python -m cs2props.lines --pickfinder  # also every book PickFinder tracks (DraftKings Pick6, Dabble, ...)

--pickfinder logs in with Selenium using the PICKFINDER_EMAIL and PICKFINDER_PASSWORD environment
variables, then reads PickFinder's own JSON API with the session token instead of screenshots + OCR.
"""
import os
import re
import sys
import time

import pandas as pd
import requests

LINES_CSV, HISTORY_CSV = "data/lines.csv", "data/lines_history.csv"
COLS = ["book", "player", "team", "opponent", "stat", "maps", "line", "over_odds", "under_odds", "start_time"]
http = requests.Session()
http.headers.update({"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
                                   "Chrome/130.0 Safari/537.36", "Accept": "application/json"})


def get(url, **params):
    r = http.get(url, params=params, timeout=30)
    r.raise_for_status()
    time.sleep(1)
    return r.json()


def market(text):
    """'MAPS 1-2 Kills' / 'kills_on_maps_1_2' / '1st_2_maps_player_headshots' -> stat='kills', maps='1-2'."""
    s = text.lower()
    stat = next((name for word, name in [("awp", "awp_kills"), ("headshot", "headshots"), ("first", "first_kills"),
                                         ("assist", "assists"), ("kill", "kills")] if word in s), s)
    digits = [int(d) for d in re.findall(r"\d", s)]
    maps = f"1-{max(digits)}" if len(digits) > 1 or "maps" in s else str(digits[0]) if digits else ""
    return dict(stat=stat, maps=maps)


def american(decimal_payout):
    m = float(decimal_payout)
    return round((m - 1) * 100) if m >= 2 else round(-100 / (m - 1))


def prizepicks():
    d = get("https://partner-api.prizepicks.com/projections", league_id=265, per_page=1000)  # 265 = CS2
    inc = {(x["type"], x["id"]): x["attributes"] for x in d.get("included", [])}
    for p in d["data"]:
        a = p["attributes"]
        if a.get("odds_type", "standard") != "standard":  # goblin/demon lines are over-only at other payouts
            continue
        pl = inc[("new_player", p["relationships"]["new_player"]["data"]["id"])]
        yield dict(book="prizepicks", player=pl["name"], team=pl.get("team"), line=a["line_score"],
                   opponent=re.sub(r"\s+MAPS?\b.*$", "", a["description"], flags=re.I),
                   start_time=a["start_time"], **market(a["stat_type"]))


def underdog():
    d = get("https://api.underdogfantasy.com/v1/over_under_lines", sport_id="CS")
    players, apps, games = ({x["id"]: x for x in d[k]} for k in ("players", "appearances", "games"))
    for ln in d["over_under_lines"]:
        if ln.get("status") != "active":
            continue
        st = ln["over_under"]["appearance_stat"]
        a = apps[st["appearance_id"]]
        p, g = players[a["player_id"]], games[a["match_id"]]
        away, home = g["full_team_names_title"].split(" vs ", 1)  # title is "AWAY vs HOME"
        team, opp = (home, away) if a["team_id"] == g["home_team_id"] else (away, home)
        # No per-pick price: standard pick'em payouts (payout_multiplier 1.0) get the same default breakeven
        # as PrizePicks. The feed's american_price (-112) is not what a pick'em entry pays.
        yield dict(book="underdog", player=" ".join(x for x in (p.get("first_name"), p.get("last_name")) if x),
                   team=team, opponent=opp, line=float(ln["stat_value"]), start_time=g["scheduled_at"],
                   **market(st["stat"]))


def sleeper():
    lines = get("https://api.sleeper.app/lines/available")
    players = get("https://api.sleeper.app/players/cs")
    games = {g["game_id"]: g.get("metadata") or {} for g in get("https://api.sleeper.app/scores/lines_game_picker")
             if g.get("sport") == "cs"}
    for x in lines:
        if x.get("sport") != "cs" or x.get("subject_type") != "player":
            continue
        o, g, p = x["options"][0], games.get(x["game_id"], {}), players.get(x["subject_id"], {})
        teams = [(g.get(side) or {}).get("name") for side in ("home_team", "away_team")]
        odds = {opt["outcome"]: american(opt["payout_multiplier"]) for opt in x["options"]}
        start = g.get("start_time_utc_s")
        yield dict(book="sleeper", player=(p.get("metadata") or {}).get("username") or p.get("username"),
                   team=o.get("subject_team"), opponent=next((t for t in teams if t and t != o.get("subject_team")), None),
                   line=o["outcome_value"], over_odds=odds.get("over"), under_odds=odds.get("under"),
                   start_time=pd.Timestamp(start, unit="s", tz="UTC").isoformat() if start else None,
                   **market(x["wager_type"]))


def pickfinder():
    from cs2props.pickfinder import current_props, main_prices, session  # Selenium login; see pickfinder.py

    with session() as get:
        for it in current_props(get):
            if it.get("isAltLine"):
                continue
            for book, app in (it.get("apps") or {}).items():
                prices = main_prices(app)  # skips goblin/demon/boosted and one-sided lines
                if book in ("prizepicks", "underdog", "sleeper") or not prices:  # those three are fetched directly
                    continue
                yield dict(book=book, player=it["player_name"], team=(it.get("team") or {}).get("name"),
                           opponent=(it.get("opponent") or {}).get("name"), line=it["line"], over_odds=prices[0],
                           under_odds=prices[1], start_time=it.get("start_date"), **market(it["market_id"]))


def main():
    rows = []
    for src in [prizepicks, underdog, sleeper] + ([pickfinder] if "--pickfinder" in sys.argv else []):
        try:
            got = list(src())
        except Exception as e:  # one book being down must not lose the others
            print(f"{src.__name__}: FAILED {e!r}", file=sys.stderr)
            continue
        print(f"{src.__name__}: {len(got)} lines")
        rows += got
    df = pd.DataFrame(rows, columns=COLS)
    df["start_time"] = pd.to_datetime(df.start_time, utc=True, format="ISO8601")
    os.makedirs("data", exist_ok=True)
    df.to_csv(LINES_CSV, index=False)
    df.assign(scraped_at=pd.Timestamp.now(tz="UTC")).to_csv(
        HISTORY_CSV, mode="a", header=not os.path.exists(HISTORY_CSV), index=False)
    print(f"wrote {len(df)} lines to {LINES_CSV}")


if __name__ == "__main__":
    main()
