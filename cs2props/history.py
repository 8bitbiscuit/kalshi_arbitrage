"""Download CS2 per-map player box scores from the public bo3.gg API into data/maps.csv.

    python -m cs2props.history               # first run: everything since the CS2 launch (about 1 hour)
    python -m cs2props.history 2025-01-01    # or start from a later date

Re-run it any time (e.g. daily before predicting): it resumes from the newest match already saved,
re-checking the last two days for late box scores. Ctrl-C is safe; only whole matches are written.
"""
import csv
import datetime as dt
import os
import sys
import time

import requests

API = "https://api.bo3.gg/api/v1/matches"
OUT = "data/maps.csv"
CS2_LAUNCH = "2023-09-27"
FIELDS = ["match_id", "start_date", "tier", "bo_type", "game_id", "map_number", "map_name", "rounds",
          "team_id", "team_name", "opp_id", "player_id", "nickname", "win",
          "kills", "deaths", "headshots", "assists", "adr", "kast"]
http = requests.Session()
http.headers["User-Agent"] = "cs2props/0.1 (personal research; ~1 request/s)"


def get(**params):
    for attempt in range(6):
        time.sleep(1)  # be polite
        r = http.get(API, params=params, timeout=120)
        if r.status_code == 200:
            return r.json()
        if r.status_code not in (429, 500, 502, 503, 504):
            r.raise_for_status()
        time.sleep(5 * 2 ** attempt)
    r.raise_for_status()


def rows(m):
    """One row per player per finished map. Uses the row-level team_id (the team at match time)."""
    games = {g["id"]: g for g in m.get("games") or [] if g.get("status") == "finished"}
    teams = {m.get("team1_id"), m.get("team2_id")}
    seen = set()
    for p in m.get("game_players") or []:
        g = games.get(p["game_id"])
        if not g or p["team_id"] not in teams or (g["id"], p["player_id"]) in seen:
            continue
        seen.add((g["id"], p["player_id"]))
        winner = (g.get("winner_team_clan") or {}).get("team_id")
        score = (g.get("winner_clan_score") or 0) + (g.get("loser_clan_score") or 0)
        yield dict(match_id=m["id"], start_date=m["start_date"], tier=m.get("tier"), bo_type=m["bo_type"],
                   game_id=g["id"], map_number=g["number"], map_name=g["map_name"],
                   rounds=g.get("rounds_count") or score or None, team_id=p["team_id"],
                   team_name=(p.get("team") or {}).get("name"), opp_id=(teams - {p["team_id"]}).pop(),
                   player_id=p["player_id"], nickname=(p.get("player") or {}).get("nickname") or p.get("player_nickname"),
                   win=None if winner is None else int(winner == p["team_id"]),
                   kills=p["kills"], deaths=p["deaths"], headshots=p["headshots"], assists=p["assists"],
                   adr=p["adr"], kast=p["kast"])


def main(since=None):
    os.makedirs("data", exist_ok=True)
    done, newest = set(), CS2_LAUNCH
    if os.path.exists(OUT):
        with open(OUT) as f:
            for r in csv.DictReader(f):
                done.add(int(r["match_id"]))
                newest = max(newest, r["start_date"])
    since = since or str(dt.date.fromisoformat(newest[:10]) - dt.timedelta(days=2))
    print(f"fetching matches after {since}; {len(done)} already saved")
    with open(OUT, "a", newline="") as f:
        out = csv.DictWriter(f, FIELDS)
        if f.tell() == 0:
            out.writeheader()
        offset = 0
        while True:
            page = get(**{"page[limit]": 50, "page[offset]": offset, "sort": "start_date", "with": "teams,games,game_players",
                          "filter[matches.status][in]": "finished", "filter[matches.discipline_id][eq]": 1,
                          "filter[matches.start_date][gt]": since})
            for m in page["results"]:
                if m["id"] in done or m.get("game_version") != 2 or None in (m.get("team1_id"), m.get("team2_id")):
                    continue
                new = list(rows(m))
                if new:  # matches without box scores yet are retried on the next run
                    out.writerows(new)
                    done.add(m["id"])
            f.flush()
            offset += len(page["results"])
            print(f"{offset}/{page['total']['count']} matches scanned, up to {page['results'][-1]['start_date'][:10]}"
                  if page["results"] else "done")
            if not page["results"] or offset >= page["total"]["count"]:
                return


if __name__ == "__main__":
    main(*sys.argv[1:])
