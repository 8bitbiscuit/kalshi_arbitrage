"""Download CS2 per-map player stats from the public bo3.gg API into data/maps.csv.

    python -m cs2props.history                # back to the CS2 launch (2023-09-27)
    python -m cs2props.history 2025-01-01     # or any earlier cut-off date

Resumable: matches already in data/maps.csv are skipped, so re-run it daily to top up.
"""
import csv
import os
import sys
import time

import requests

API = "https://api.bo3.gg/api/v1"
OUT = "data/maps.csv"
FIELDS = ["match_id", "start_date", "tier", "bo_type", "game_id", "map_number", "map_name", "rounds",
          "team_id", "team_name", "opp_id", "opp_name", "player_id", "nickname", "win",
          "kills", "deaths", "headshots", "assists", "adr", "kast", "rating"]
session = requests.Session()
session.headers["User-Agent"] = "cs2props-research/0.1 (personal, 1 req/s)"


def get(path, **params):
    for attempt in range(6):
        time.sleep(1)  # be polite: ~1 request per second
        r = session.get(f"{API}/{path}", params=params, timeout=30)
        if r.status_code == 200:
            return r.json()
        if r.status_code not in (429, 500, 502, 503, 504):
            r.raise_for_status()
        time.sleep(2 ** attempt * 5)
    r.raise_for_status()


def map_rows(match, game):
    t1, t2 = match.get("team1") or {}, match.get("team2") or {}
    rounds = game.get("rounds_count") or ((game.get("winner_clan_score") or 0) + (game.get("loser_clan_score") or 0))
    for p in get(f"games/{game['id']}/players_stats"):
        team = (p.get("team_clan") or {}).get("team") or {}
        player = (p.get("steam_profile") or {}).get("player") or {}
        team_id = str(team.get("id") or f"name:{p['clan_name']}")
        opp = t2 if str(t1.get("id")) == team_id else t1
        yield dict(match_id=match["id"], start_date=match["start_date"], tier=match.get("tier"),
                   bo_type=match["bo_type"], game_id=game["id"], map_number=game["number"],
                   map_name=game["map_name"], rounds=rounds or None, team_id=team_id,
                   team_name=team.get("name") or p["clan_name"],
                   opp_id=str(opp.get("id") or f"name:{p.get('enemy_clan_name')}"),
                   opp_name=opp.get("name") or p.get("enemy_clan_name"),
                   player_id=player.get("id") or f"steam:{p['steam_profile_id']}",
                   nickname=player.get("nickname") or (p.get("steam_profile") or {}).get("nickname"),
                   win=p["win"], kills=p["kills"], deaths=p["death"], headshots=p["headshots"],
                   assists=p["assists"], adr=p["adr"], kast=p["kast"], rating=p["player_rating"])


def main(since="2023-09-27"):
    os.makedirs("data", exist_ok=True)
    done = set()
    if os.path.exists(OUT):
        with open(OUT) as f:
            done = {int(r["match_id"]) for r in csv.DictReader(f)}
    new_file = not os.path.exists(OUT)
    with open(OUT, "a", newline="") as f:
        out = csv.DictWriter(f, FIELDS)
        if new_file:
            out.writeheader()
        offset = 0
        while True:
            page = get("matches", **{
                "page[limit]": 50, "page[offset]": offset, "sort": "-start_date", "with": "teams,games",
                "filter[matches.status][in]": "finished", "filter[matches.parsed_status][eq]": "done",
                "filter[matches.discipline_id][eq]": 1})
            for m in page["results"]:
                if m["start_date"] < since:
                    return print(f"reached {since}; {len(done)} matches in {OUT}")
                if m["id"] in done or m.get("game_version") != 2:
                    continue
                rows = [r for g in sorted(m.get("games") or [], key=lambda g: g["number"])
                        if g.get("status") == "finished" for r in map_rows(m, g)]
                out.writerows(rows)  # whole match at once, so an interrupted run never leaves half a match
                f.flush()
                done.add(m["id"])
                print(f"{m['start_date'][:10]} {m.get('slug')}: {len(rows)} rows")
            offset += len(page["results"])
            if not page["results"] or offset >= page["total"]["count"]:
                return print(f"end of results; {len(done)} matches in {OUT}")


if __name__ == "__main__":
    main(*sys.argv[1:])
