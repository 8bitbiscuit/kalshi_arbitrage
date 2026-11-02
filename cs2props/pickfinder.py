"""PickFinder (needs your login): prop history for backtesting, plus a helper that shows its API calls.

    python -m cs2props.pickfinder history            # data/prop_history.csv -> python -m cs2props.model backtest
    python -m cs2props.pickfinder discover KaiR0N    # save the data requests a player page makes, for debugging

Set PICKFINDER_EMAIL and PICKFINDER_PASSWORD; a Firefox window opens and logs in.

What is known about PickFinder's API:
  - GET /props?sport=cs2 lists current props, which is the only way found to get player ids.
  - GET /players/{id}/gamelog has every series with per-map stats (the results). For some sports it
    also has `lines` with the archived line, but a logged-in CS2 gamelog has none (checked 2026-11-02).
  - GET /players/{id}/props returns only the player's current props, not past ones (checked 2026-11-02).
So the CS2 "Prop History" panel loads its lines from another request. `discover` records every
PickFinder data request a player page makes, so that request and its response can be found.
"""
import json
import os
import re
import sys
import time
from contextlib import contextmanager

import pandas as pd

from cs2props.lines import COLS, http
from cs2props.model import team_key

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
    """Series with 2+ maps where the player played every map, so periodStats[:2] really are maps 1 and 2
    (the gamelog leaves out maps a player sat out)."""
    return [{**{stat: sum(m.get(stat) or 0 for m in g["periodStats"][:2]) for stat in MARKETS.values()},
             "start": pd.to_datetime(g["startTime"], utc=True), "team": g.get("team"), "vs": g.get("vs"),
             "abbr": (g.get("vsAbbr") or "").upper(), "lines": g.get("lines") or {}}
            for g in games if len(g.get("periodStats") or []) >= 2
            and len(g["periodStats"]) == sum(x.get("score") or 0 for x in g.get("scores") or [])]


def lookup(results, start, opponent):
    """The prop's own series: same opponent (abbreviation, else normalised name), starting between 3 h before
    and 12 h after the listed time (matches start late, rarely early); the nearest one if several."""
    abbr, name = (opponent.get("abbreviation") or "").upper(), team_key(opponent.get("name"))
    same = [r for r in results if pd.Timedelta(hours=-3) <= r["start"] - start <= pd.Timedelta(hours=12)
            and ((abbr and abbr == r["abbr"]) or name == team_key(r["vs"]))]
    return min(same, key=lambda r: abs(r["start"] - start)) if same else None


def main_prices(app):
    """(over, under) prices of a book's main line; None for goblin/demon/boosted or one-sided lines."""
    out = app.get("outcomes") or {}
    sides = [out.get("over"), out.get("under")]
    if all(sides) and all(side.get("modifier") in (None, "", "none") for side in sides):
        return sides[0].get("price"), sides[1].get("price")
    return None


def player_rows(it, games, props, since):
    """Prop-history rows (backtest schema + actual) for one player from their gamelog and props responses."""
    name, results, rows = it["player_name"], series_results(games), []
    for r in results:  # lines archived in the gamelog itself
        if r["start"] >= since:
            for display, stat in MARKETS.items():
                if r["lines"].get(display) is not None:
                    rows.append(dict(book="pickfinder", player=name, team=r["team"], opponent=r["vs"], stat=stat,
                                     maps="1-2", line=r["lines"][display], start_time=r["start"].isoformat(),
                                     actual=r[stat], source="gamelog"))
    now = pd.Timestamp.now(tz="UTC")
    for p in props:  # past prop items, one row per book offering the main line
        if p.get("stat") not in MARKETS or p.get("isAltLine") or not p.get("start_date"):
            continue
        start, opp = pd.to_datetime(p["start_date"], utc=True), p.get("opponent") or {}
        r = lookup(results, start, opp)
        if not (since <= start < now) or r is None:  # result not logged yet: picked up by a later run
            continue
        for book, app in (p.get("apps") or {}).items():
            prices = main_prices(app)
            if prices:
                rows.append(dict(book=book, player=name, team=(p.get("team") or {}).get("name") or r["team"],
                                 opponent=opp.get("name"), stat=MARKETS[p["stat"]], maps="1-2", line=p["line"],
                                 over_odds=prices[0], under_odds=prices[1], start_time=start.isoformat(),
                                 actual=r[MARKETS[p["stat"]]], source="props"))
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
    if os.path.exists(PROPS_CSV):  # keep earlier runs' lines (other players, older dates); new rows win
        df = pd.concat([df, pd.read_csv(PROPS_CSV)]).drop_duplicates(["book", "player", "stat", "start_time", "line"])
    df.to_csv(PROPS_CSV, index=False)
    print(f"{len(df)} past lines in {PROPS_CSV}; next: python -m cs2props.model backtest")
    if not rows:
        print("No past lines were in the gamelog or props responses. Run `python -m cs2props.pickfinder discover "
              "<player>` and look in data/pickfinder_debug/ for the response that holds them.")


# Runs in the page: keeps the URL and response of every fetch/XHR from now on, and lifts the browser's
# 250-entry request log so later requests aren't dropped from it.
RECORD_JS = """
performance.setResourceTimingBufferSize(100000);
window.__pf = window.__pf || [];
if (!window.__pfHooked) {
  window.__pfHooked = true;
  const keep = (url, status, body) => window.__pf.push({url: new URL(url, location.href).href, status, body: String(body)});
  const fetch0 = window.fetch;
  window.fetch = async (...args) => {
    const res = await fetch0(...args);
    res.clone().text().then(t => keep((args[0] && args[0].url) || args[0], res.status, t)).catch(() => {});
    return res;
  };
  const open0 = XMLHttpRequest.prototype.open, send0 = XMLHttpRequest.prototype.send;
  XMLHttpRequest.prototype.open = function (method, url, ...rest) { this.__url = url; return open0.call(this, method, url, ...rest); };
  XMLHttpRequest.prototype.send = function (...args) {
    this.addEventListener("load", () => { try { keep(this.__url, this.status, this.responseText); } catch (e) {} });
    return send0.apply(this, args);
  };
}
"""


def pickfinder_data(url):
    """PickFinder's own data requests (API or website routes), not the login service, analytics or assets."""
    host = re.sub(r"^https?://([^/]+).*", r"\1", url)
    return (host == "pickfinder.app" or host.endswith(".pickfinder.app")) and not host.startswith("clerk.")


def discover(player):
    """Open a player's page, let you open Prop History, then save every PickFinder data response it made."""
    os.makedirs(DEBUG_DIR, exist_ok=True)
    with session() as get:
        key = re.sub(r"[^a-z0-9]", "", player.lower())
        pid = next((it["player_id"] for it in current_props(get)
                    if re.sub(r"[^a-z0-9]", "", it["player_name"].lower()) == key), player)
        driver = get.driver
        driver.get(f"{SITE}/players/cs2/{pid}")
        driver.execute_script(RECORD_JS)
        input("In the Firefox window: choose 'Maps 1-2 Kills', open the Prop History panel and wait until its\n"
              "cards show. Then press Enter here. ")
        calls = {c["url"]: c for c in driver.execute_script("return window.__pf || []")}
        # Requests made while the page first loaded, before recording began: fetch them again.
        loaded = driver.execute_script("return performance.getEntriesByType('resource').filter(e => "
                                       "['fetch', 'xmlhttprequest'].includes(e.initiatorType)).map(e => e.name)")
        for url in dict.fromkeys(loaded):
            if url not in calls and pickfinder_data(url):
                try:
                    body = json.dumps(get(url)) if url.startswith(API) else driver.execute_async_script(
                        "const done = arguments[1]; fetch(arguments[0], {credentials: 'include'})"
                        ".then(r => r.text()).then(done).catch(e => done('ERROR ' + e))", url)
                except Exception as e:
                    body = f"ERROR {e!r}"
                calls[url] = {"url": url, "status": None, "body": body}
        for old in os.listdir(DEBUG_DIR):  # don't mix in files from an earlier run
            if old.startswith("call_"):
                os.remove(os.path.join(DEBUG_DIR, old))
        for i, c in enumerate(c for c in calls.values() if pickfinder_data(c["url"])):
            try:
                c["body"] = json.loads(c["body"])
            except (TypeError, ValueError):
                pass
            with open(f"{DEBUG_DIR}/call_{i:02d}.json", "w") as f:
                json.dump(c, f)
            text = json.dumps(c["body"])
            has_line = '"line' in text
            print(f"call_{i:02d}.json  {len(text):>8} bytes  has 'line': {has_line!s:5}  {c['url']}")
    print("Send the call_*.json whose response holds the past lines (look for 'line' and the actual results).")


if __name__ == "__main__":
    cmd, *args = sys.argv[1:] or ["history"]
    {"history": history, "discover": discover}[cmd](*args)
