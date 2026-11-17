"""PickFinder (needs your login): prop history for backtesting, plus a helper that records its requests.

    python -m cs2props.pickfinder history            # data/prop_history.csv -> python -m cs2props.model backtest
    python -m cs2props.pickfinder discover KaiR0N    # save the data requests a player page makes, for debugging

Set PICKFINDER_EMAIL and PICKFINDER_PASSWORD; a Firefox window opens and logs in.

Where the data comes from (found with `discover` on 2026-11-02):
  - api-v2 GET /props?sport=cs2&moneylines=1 lists current props: player ids, names and OpticOdds ids
    (player_oo_id), the fixture_id and, with moneylines=1 (as PickFinder's own /props/favorites call sends it),
    the market's moneyline: team_ml / opponent_ml (Thunderpick) and team_ml_percent / opponent_ml_percent (the
    same with the vig removed). api-v2 GET /players/{player_id}/props, a player's props, always carries them.
  - api-v2 GET /players/{player_id}/gamelog has every series with per-map stats: the results.
  - www GET /api/players/{player_oo_id}/props/historical?stat=Maps 1-2 Kills is the Prop History panel:
    one entry per past match with the line, each book's prices and the market's moneyline as of then:
    teamOdds {implied, american} and opponentOdds {implied}, implied with the vig removed (0.3882 + 0.6118).
    Its matchKey is "<scheduled US Eastern date>-<opponent id>", and the opponent id is the gamelog's vsId.
    It is a website route, so it is called from inside the logged-in page (it uses cookies, not a token).
  - www GET /api/cs2/matches/{fixture_id}/odds?map=1 is a match's odds: Thunderpick's series moneyline, map
    handicap, total maps, correct scores and the map-1 moneyline (unused: the props carry the moneyline).
Checked against real responses on 2026-11-04 and 11-05.
"""
import json
import os
import re
import sys
import time
from contextlib import contextmanager
from urllib.parse import quote, urlencode

import numpy as np
import pandas as pd

from cs2props.lines import COLS, http
from cs2props.model import american_to_prob

API, SITE = "https://api-v2.pickfinder.app", "https://www.pickfinder.app"
MARKETS = {"Maps 1-2 Kills": "kills", "Maps 1-2 HS": "headshots"}  # PickFinder's display names
PROPS_CSV, DEBUG_DIR = "data/prop_history.csv", "data/pickfinder_debug"


@contextmanager
def session():
    """Log in with Firefox. Yields get(path_or_url, **params) for the api-v2 API (sends a fresh session
    token), with get.page(path_or_url, **params) for www.pickfinder.app routes and get.driver."""
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

        def page(path, **params):
            time.sleep(1)
            url = (path if path.startswith("http") else SITE + path) + (
                "?" + urlencode(params, quote_via=quote) if params else "")
            status, text = driver.execute_async_script(
                "const done = arguments[1]; fetch(arguments[0], {credentials: 'include'})"
                ".then(async r => done([r.status, await r.text()])).catch(e => done([0, String(e)]))", url)
            if status != 200:
                raise RuntimeError(f"HTTP {status} from {url}: {text[:200]}")
            return json.loads(text)

        get.page, get.driver = page, driver
        yield get
    finally:
        driver.quit()


def current_props(get):
    """Every current CS2 prop item, all pages."""
    page = 1
    while True:
        d = get("/props", sport="cs2", page=page, perPage=100, moneylines=1)
        yield from d.get("items", [])
        if not d.get("items") or page >= d.get("totalPages", 1):
            return
        page += 1


def series_win_prob(item):
    """The market's chance that a props or Prop History item's team wins the series, or NaN: Thunderpick's
    moneyline with the vig removed (props: team_ml_percent, or team_ml and opponent_ml; history: teamOdds)."""
    p = item.get("team_ml_percent") or (item.get("teamOdds") or {}).get("implied")
    if not p and item.get("team_ml") and item.get("opponent_ml"):
        team, opp = american_to_prob(item["team_ml"]), american_to_prob(item["opponent_ml"])
        p = team / (team + opp)
    return p if p and 0 < p < 1 else np.nan


def map_win_prob(series_p):
    """The market's chance that a team wins a map, from its chance to win the series, so it means what the
    model's Elo win_prob means; NaN when missing. Maps taken as independent, so a series win is p^2 (3 - 2p).
    ponytail: every series taken as best of 3 (props don't say); a best of 5 would come out a little too lopsided."""
    if not 0 < (series_p or 0) < 1:
        return np.nan
    p = np.linspace(0, 1, 1001)
    return float(np.interp(series_p, p * p * (3 - 2 * p), p))


def series_results(games):
    """Series with 2+ maps where the player played every map, so periodStats[:2] really are maps 1 and 2
    (the gamelog leaves out maps a player sat out)."""
    out = []
    for g in games:
        maps = g.get("periodStats") or []
        if len(maps) >= 2 and len(maps) == sum(x.get("score") or 0 for x in g.get("scores") or []):
            start = pd.to_datetime(g["startTime"], utc=True)
            out.append({**{stat: sum(m.get(stat) or 0 for m in maps[:2]) for stat in MARKETS.values()},
                        "start": start, "et_date": start.tz_convert("America/New_York").date(),
                        "team": g.get("team"), "vs": g.get("vs"), "vs_id": str(g.get("vsId"))})
    return out


def book_line(app, default):
    """A book's own line: PrizePicks links carry it (".../?projections=16194603-o-28.5"); others use the entry's."""
    link = ((app.get("outcomes") or {}).get("over") or {}).get("deep_link") or {}
    found = re.search(r"-o-(\d+(?:\.\d+)?)$", str(link.get("desktop") or link.get("ios") or ""))
    return float(found.group(1)) if found else default


def main_prices(app):
    """(over, under) prices of a book's main line; None for goblin/demon/boosted or one-sided lines."""
    out = app.get("outcomes") or {}
    sides = [out.get("over"), out.get("under")]
    if all(sides) and all(side.get("modifier") in (None, "", "none") for side in sides):
        return sides[0].get("price"), sides[1].get("price")
    return None


def player_rows(it, games, history, since):
    """Backtest rows for one player: every past main line (one row per book) with its series' result.

    history maps a market name to its /props/historical response. An entry's series is the gamelog series
    against the same opponent id within a day of the scheduled date. Entries with no such series (not played
    yet, missing, the player sat out a map) or with two equally close ones are skipped, not guessed."""
    results, rows = series_results(games), []
    for display, entries in history.items():
        for h in entries:
            day, opp_id = h["matchKey"].rsplit("-", 1)
            day = pd.Timestamp(day).date()
            gaps = {id(r): abs((r["et_date"] - day).days) for r in results if r["vs_id"] == opp_id}
            near = [r for r in results if gaps.get(id(r), 9) <= 1 and gaps[id(r)] == min(gaps.values())]
            if day < since or len(near) != 1:
                continue
            r = near[0]
            for book, app in (h.get("apps") or {}).items():
                prices = main_prices(app)
                if prices:
                    rows.append(dict(book=book, player=it["player_name"], team=r["team"], opponent=r["vs"],
                                     stat=MARKETS[display], maps="1-2", line=book_line(app, h["line"]),
                                     over_odds=prices[0], under_odds=prices[1], start_time=r["start"].isoformat(),
                                     actual=r[MARKETS[display]], market_win_prob=map_win_prob(series_win_prob(h))))
    return rows


def history(months=6):
    since = (pd.Timestamp.now(tz="UTC") - pd.DateOffset(months=int(months))).date()
    rows = []
    with session() as get:
        # ponytail: only players with a prop up right now; re-run on other days to cover more players.
        players = {it["player_id"]: it for it in current_props(get)}
        print(f"{len(players)} CS2 players with current props")
        for n, (pid, it) in enumerate(players.items(), 1):
            try:
                games = get(f"/players/{pid}/gamelog")
                games = games if isinstance(games, list) else games.get("games", [])
                hist = {d: get.page(f"/api/players/{it['player_oo_id']}/props/historical", stat=d) for d in MARKETS}
                new = player_rows(it, games, hist, since)
            except Exception as e:  # one bad player must not lose the rest
                print(f"{it['player_name']}: FAILED {e!r}", file=sys.stderr)
                continue
            rows += new
            print(f"{n}/{len(players)} {it['player_name']}: {sum(map(len, hist.values()))} past lines listed, "
                  f"{len(new)} rows with results")
    key = ["book", "player", "stat", "start_time", "line"]
    df = pd.DataFrame(rows, columns=COLS + ["actual"]).drop_duplicates(key)
    print(f"market moneyline on {df.market_win_prob.notna().sum()} of {len(df)} past lines (the rest use our Elo)")
    if os.path.exists(PROPS_CSV):  # keep earlier runs' lines (other players, older dates); new rows win
        df = pd.concat([df, pd.read_csv(PROPS_CSV)]).drop_duplicates(key)
    df.to_csv(PROPS_CSV, index=False)
    print(f"{len(df)} past lines in {PROPS_CSV}; next: python -m cs2props.model backtest")


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
                    body = json.dumps(get(url) if url.startswith(API) else get.page(url))
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
