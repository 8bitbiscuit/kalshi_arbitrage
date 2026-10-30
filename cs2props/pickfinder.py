"""PickFinder (needs your login): prop history for backtesting, plus a helper that shows its API calls.

    python -m cs2props.pickfinder history            # data/prop_history.csv -> python -m cs2props.model backtest
    python -m cs2props.pickfinder discover KaiR0N    # save the API calls a player page makes, for debugging

Set PICKFINDER_EMAIL and PICKFINDER_PASSWORD; a Firefox window opens and logs in.

What PickFinder's public site code shows (this could not be run here without a login):
  - GET /props?sport=cs2 lists current props, which is the only way found to get player ids.
  - GET /players/{id}/gamelog has every series with per-map stats (the results). For some sports it
    also has `lines`, keyed by market name ("Maps 1-2 Kills", "Maps 1-2 HS"), with the archived line.
  - GET /players/{id}/props may also return past props (line, each book's prices, moneyline %).
The CS2 "Prop History" panel's own request lives in code behind the login. If `history` finds no
past lines, run `discover` once and look in data/pickfinder_debug/ for the response holding them.
"""
import json
import os
import re
import sys
import time
from contextlib import contextmanager

import pandas as pd

from cs2props.lines import COLS, http

API, SITE = "https://api-v2.pickfinder.app", "https://www.pickfinder.app"
MARKETS = {"Maps 1-2 Kills": "kills", "Maps 1-2 HS": "headshots"}  # PickFinder's display names
PROPS_CSV, DEBUG_DIR = "data/prop_history.csv", "data/pickfinder_debug"


@contextmanager
def session():
    """Log in with Firefox and yield get(path_or_url, **params), which sends a fresh session token."""
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.support.ui import WebDriverWait

    driver = webdriver.Firefox()
    try:
        wait = WebDriverWait(driver, 30)
        driver.get(f"{SITE}/props")
        wait.until(EC.presence_of_element_located((By.NAME, "identifier"))).send_keys(
            os.environ["PICKFINDER_EMAIL"], Keys.ENTER)
        wait.until(EC.element_to_be_clickable((By.NAME, "password"))).send_keys(
            os.environ["PICKFINDER_PASSWORD"], Keys.ENTER)
        wait.until(lambda d: d.get_cookie("__session"))

        def get(path, **params):
            time.sleep(1)  # be polite
            token = driver.get_cookie("__session")["value"]  # short-lived; the open page keeps refreshing it
            r = http.get(path if path.startswith("http") else API + path, params=params, timeout=30,
                         headers={"Authorization": f"Bearer {token}"})
            r.raise_for_status()  # a rejected token must fail loudly, not look like "no data"
            return r.json()

        get.driver = driver
        yield get
    finally:
        driver.quit()


def current_props(get):
    """Every current CS2 prop item, all pages."""
    page = 1
    while True:
        d = get("/props", sport="cs2", page=page, perPage=100)
        yield from d.get("items", [])
        if not d.get("items") or page >= d.get("totalPages", 1):
            return
        page += 1


def series_results(games):
    """Series with 2+ maps: start time, opponent, maps 1-2 kills/headshots, and any archived lines."""
    return [{**{stat: sum(m.get(stat) or 0 for m in g["periodStats"][:2]) for stat in MARKETS.values()},
             "start": pd.to_datetime(g["startTime"], utc=True), "vs": g.get("vs"),
             "abbr": str(g.get("vsAbbr", "")).upper(), "lines": g.get("lines") or {}}
            for g in games if len(g.get("periodStats") or []) >= 2]


def lookup(results, start, opp_abbr):
    """The series within 36 h of the prop's start (book and PickFinder times differ): same opponent
    abbreviation first, else the only series in that window."""
    near = [r for r in results if abs(r["start"] - start) <= pd.Timedelta(hours=36)]
    same = [r for r in near if r["abbr"] == str(opp_abbr).upper()]
    return same[0] if same else near[0] if len(near) == 1 else None


def player_rows(it, games, props, since):
    """Prop-history rows (backtest schema + actual) for one player from their gamelog and props responses."""
    name, team = it["player_name"], (it.get("team") or {}).get("name")
    results, rows = series_results(games), []
    for r in results:  # lines archived in the gamelog itself
        if r["start"] >= since:
            for display, stat in MARKETS.items():
                if r["lines"].get(display) is not None:
                    rows.append(dict(book="pickfinder", player=name, team=team, opponent=r["vs"], stat=stat, maps="1-2",
                                     line=r["lines"][display], start_time=r["start"].isoformat(), actual=r[stat],
                                     source="gamelog"))
    now = pd.Timestamp.now(tz="UTC")
    for p in props:  # past prop items, one row per book
        if p.get("stat") not in MARKETS or p.get("isAltLine") or not p.get("start_date"):
            continue
        start = pd.to_datetime(p["start_date"], utc=True)
        r = lookup(results, start, (p.get("opponent") or {}).get("abbreviation"))
        if not (since <= start < now) or r is None:
            continue
        for book, app in (p.get("apps") or {}).items():
            out = app.get("outcomes") or {}
            rows.append(dict(book=book, player=name, team=team, opponent=(p.get("opponent") or {}).get("name"),
                             stat=MARKETS[p["stat"]], maps="1-2", line=p["line"],
                             over_odds=(out.get("over") or {}).get("price"), under_odds=(out.get("under") or {}).get("price"),
                             start_time=start.isoformat(), actual=r[MARKETS[p["stat"]]], source="props"))
    return rows


def history(months=6):
    since = pd.Timestamp.now(tz="UTC") - pd.DateOffset(months=int(months))
    os.makedirs(DEBUG_DIR, exist_ok=True)
    rows = []
    with session() as get:
        # ponytail: only players with a prop up right now; re-run on other days to cover more players.
        players = {it["player_id"]: it for it in current_props(get)}
        print(f"{len(players)} CS2 players with current props")
        for n, (pid, it) in enumerate(players.items(), 1):
            try:
                games = get(f"/players/{pid}/gamelog")
                props = get(f"/players/{pid}/props")
            except Exception as e:  # one bad player must not lose the rest
                print(f"{it['player_name']}: FAILED {e!r}", file=sys.stderr)
                continue
            if n == 1:  # keep one raw example of each response for debugging
                for nm, body in (("gamelog", games), ("props", props)):
                    with open(f"{DEBUG_DIR}/example_{nm}.json", "w") as f:
                        json.dump(body, f)
            games = games if isinstance(games, list) else games.get("games", [])
            props = props.get("items", []) if isinstance(props, dict) else props
            new = player_rows(it, games, props, since)
            rows += new
            print(f"{n}/{len(players)} {it['player_name']}: {len(games)} games, {len(new)} past lines")
    cols = COLS + ["actual", "source"]
    df = pd.DataFrame(rows, columns=cols).drop_duplicates(["book", "player", "stat", "start_time", "line"])
    if os.path.exists(PROPS_CSV):  # keep lines from earlier runs (other players, older dates)
        df = pd.concat([pd.read_csv(PROPS_CSV), df]).drop_duplicates(["book", "player", "stat", "start_time", "line"])
    df.to_csv(PROPS_CSV, index=False)
    print(f"{len(df)} past lines in {PROPS_CSV}; next: python -m cs2props.model backtest")
    if not rows:
        print("No past lines were in the gamelog or props responses. Run `python -m cs2props.pickfinder discover "
              "<player>` and look in data/pickfinder_debug/ for the response that holds them.")


def discover(player, seconds=90):
    """Open a player's page, let you click its Prop History, then save every API response it made."""
    os.makedirs(DEBUG_DIR, exist_ok=True)
    with session() as get:
        key = re.sub(r"[^a-z0-9]", "", player.lower())
        pid = next((it["player_id"] for it in current_props(get)
                    if re.sub(r"[^a-z0-9]", "", it["player_name"].lower()) == key), player)
        get.driver.get(f"{SITE}/players/cs2/{pid}")
        print(f"In the Firefox window, pick 'Maps 1-2 Kills' and open Prop History. Recording for {seconds} s ...")
        time.sleep(int(seconds))
        urls = get.driver.execute_script(
            "return [...new Set(performance.getEntriesByType('resource').map(e => e.name))]")
        for i, url in enumerate(u for u in urls if u.startswith(API)):
            try:
                body = get(url)
            except Exception as e:
                body = {"error": repr(e)}
            with open(f"{DEBUG_DIR}/call_{i:02d}.json", "w") as f:
                json.dump({"url": url, "body": body}, f)
            text = json.dumps(body)
            has_line = '"line' in text
            print(f"call_{i:02d}.json  {len(text):>8} bytes  has 'line': {has_line}  {url}")


if __name__ == "__main__":
    cmd, *args = sys.argv[1:] or ["history"]
    {"history": history, "discover": discover}[cmd](*args)
