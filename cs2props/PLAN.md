# CS2 player-prop model: plan and how to run it

Goal: for each current "Maps 1-2 Kills" / "Maps 1-2 Headshots" line (PrizePicks, Underdog, Sleeper,
optionally every book PickFinder tracks), predict whether the player goes over or under, using a
decision tree trained on CS2 match history.

```
bo3.gg API ──history.py──▶ data/maps.csv ──model.py train──▶ data/model_*.joblib
                                                              │
DFS / PickFinder ──lines.py──▶ data/lines.csv ──model.py predict──▶ data/picks.csv
                         └──▶ data/lines_history.csv  (archive for grading real-line accuracy later)
```

## Run it

```bash
pip install -r cs2props/requirements.txt
python -m cs2props.history          # first run about 1 hour (all of CS2); later runs take a minute
python -m cs2props.model train      # prints honest test metrics, saves the two trees
python -m cs2props.lines            # current lines from PrizePicks, Underdog, Sleeper (no login)
python -m cs2props.model predict    # prints and saves picks, best edge first
python -m pytest cs2props -q        # self-checks (leakage, probabilities, name matching, end to end)
```

Daily: `history` then `lines` then `predict`. Retrain weekly or after big roster moves.

## 1. Training data: bo3.gg, not HLTV

| Source | Has per-map kills + HS? | Access | Verdict |
|---|---|---|---|
| **bo3.gg API** (`api.bo3.gg/api/v1`) | Yes, plus ADR, KAST, map, round score, winner, both team ids | Open JSON, no key, about 1 req/s is fine | **Primary.** About 77k finished CS matches (2020 onward), 150k maps, tiers S to D including the tier-C Brazilian events in the screenshots |
| HLTV.org | Yes (`/stats/matches/mapstatsid/...`) | Cloudflare challenge; Terms §2.2 ban scraping | Fallback only, from a home connection with a headed browser (SeleniumBase CDP mode), 4-8 s between pages |
| PickFinder | Only each player's last few results | Login, paid | Not a training source |

What `history.py` pulls: `GET /matches?with=teams,games,game_players&sort=start_date&page[limit]=50`, which
returns 50 matches with every map and every player's box score in one request. That is about 1,500
requests for all of CS2 (since 2023-09-27), versus about 150,000 for the per-map endpoint. One row per
player per map goes to `data/maps.csv`.

Checked against real settlement: Sleeper's own "last 10 results" for 165 player-matches equalled the
bo3.gg sum over maps 1 and 2 exactly, overtime included.

Data problems found in the real download, and how the code handles them:

| Problem | How common | Handling |
|---|---|---|
| Rows with no headshots/ADR/KAST are partial live snapshots, so kills run about 20% low (5.45 vs 6.68 total kills per round). Example: one map with 10 kills in total here vs 137 in the full scoreboard | About 20% of maps before March 2026, none since | Dropped. A new match is only saved once every map is complete |
| Headshots larger than kills (values like 810) | 31 maps on 2025-10-27/28 | Map dropped |
| Fewer than 13 rounds (forfeit or abandoned) | About 110 maps | Map dropped |
| Rows with no player id | 36 rows | Dropped |
| Map order in `games[]` is reversed, and the nested `player.team_id` is today's team | Always | Sorted by `number`; the row-level `team_id` is used |
| Box scores arrive late | Common for fresh matches | Each run re-checks the last two days |

## 2. Features (all computed strictly before the match starts)

The same things PickFinder shows in the screenshots, built so nothing from the predicted match leaks in:

| Feature | Meaning | Screenshot equivalent |
|---|---|---|
| `kills_l5`, `kills_l20`, `headshots_l20` | Average per map over the last 5 / 20 maps | L5 / L10 / L15 averages |
| `kpr`, `rounds_l20` | Kills per round; average rounds per map (close games mean more rounds, more kills) | KPR in Map Stats |
| `hs_pct` | Headshots / kills over the last 20 maps | HS % |
| `adr_l20`, `kast_l20` | Damage per round, KAST | K/D, rating |
| `map_factor` | Player's KPR on each map (shrunk toward overall KPR with a 100-round prior), weighted by how often **both teams** played each map in their last 15 maps, divided by overall KPR | Map Picks + per-map Kills/KPR |
| `win_prob`, `elo_opp` | Map-level Elo built from every bo3.gg result; win probability for the player's team and opponent strength. A team id seen for the first time (a roster moving org, e.g. LP to BARCA) starts from its players' previous teams' ratings | Moneyline implied win % |
| `maps_played` | Experience; lets the tree distrust thin histories | sample size |

Why map pool, not actual maps: lines lock before the veto, and bo3.gg has had no veto data since
mid-September 2026. Using the actual maps in training but not at prediction time would teach the tree
something it can't use live.

Why our own Elo instead of the sportsbook moneyline: there is no history of pre-match odds (bo3.gg only
keeps the last, post-result price), so a book price can't be a training feature. Elo is computed the same
way in training and live.

## 3. Model

- **Target:** maps 1-2 kills (and a second tree for headshots), for every player in every BO3/BO5 that
  reached map 2. A *regression* tree, because there are no historical lines to label over/under.
- **Over/under probability:** each validation result gives a held-out error `z = (actual − predicted) / √predicted`.
  The z values are centred so a level shift between periods doesn't bias every pick. P(over 30.5) is
  the share of z values for which `projection + z·√projection` lands above 30.5. Counting stats spread
  like √mean, so one z distribution fits 15-kill and 35-kill players alike. Whole-number lines
  (PrizePicks posts 35, 27, ...) get push handling.
- **Tuning:** `max_depth` 3-10 and `min_samples_leaf` 25-400, chosen on a validation slice by MAE.
- **Split by time:** oldest 70% train, next 15% validate (tuning and the error spread), newest 15% test.
- **What `train` prints:** test MAE of the tree versus the naive "2 × 20-map average", a stand-in-line
  hit rate (optimistic, since real books are sharper than a 20-map average), Brier score versus a coin
  flip, and the top features.

### Results on real data

Trained on bo3.gg data from July 2025 to October 2026, about 90,000 player-series. Tested on the newest
15% (2026-09-16 to 2026-10-30), which the model never saw:

| | Kills, maps 1-2 | Headshots, maps 1-2 |
|---|---|---|
| Tree mean absolute error | 6.09 | 3.87 |
| Naive "2 × 20-map average" | 6.34 | 3.97 |
| Brier score on the stand-in line (0.250 = coin flip) | 0.237 | 0.241 |
| Most important features | kills per round, win probability, map factor, opponent Elo | 20-map headshots, HS% |

On the 2026-10-30 slate (156 lines), projections averaged within 0.03 kills and 0.21 headshots of the
books' lines (standard deviations 1.5 and 1.4). The model agrees with the books on level, so any edge
lives in disagreements of 1-2 kills. That is also why the stand-in hit rate (about 60%) is no guide to
real profit.

A single tree also gives every player in the same leaf an identical projection. Gradient boosting
(next steps) removes that coarseness.

## 4. Current lines

All verified live on 2026-10-30, no login or browser needed:

| Book | Endpoint | Notes |
|---|---|---|
| PrizePicks | `partner-api.prizepicks.com/projections?league_id=265` | Main `api.prizepicks.com` is DataDome-blocked; the partner host is open. Standard lines only (goblin/demon are over-only). No price, so breakeven defaults to 54.3% |
| Underdog | `api.underdogfantasy.com/v1/over_under_lines?sport_id=CS` | Standard pick'em payout, so the same default breakeven as PrizePicks. The feed's `-112` tag is not what a pick'em entry pays |
| Sleeper | `api.sleeper.app/lines/available` + `/players/cs` + `/scores/lines_game_picker` | Payout multipliers converted to American odds (1.78 → -128, as in the screenshot) |
| PickFinder (optional) | `api-v2.pickfinder.app/props?sport=cs2` with the logged-in `__session` token | Replaces the screenshot + OCR. Adds DraftKings Pick6, Dabble, ParlayPlay, ThunderPick, and others. **Untested**, needs your login |

Every scrape is appended to `data/lines_history.csv`.

## 5. Prediction

`predict` keeps the maps 1-2 kills/headshots lines, then matches each to bo3.gg:

- **Teams:** exact match after dropping words like Team, Esports, Gaming and CS ("Team Vitality" → "vitality"). Fuzzy team matching was wrong every time it fired in a two-month replay ("PRIVATE" → "Privateer"), so an unknown team stays unknown.
- **Players:** exact name on that team first, then exact name anywhere (transfers), then a close name
  on that team (Sleeper's "MaxOff" is bo3.gg's "max"). When the matched player's last team differs
  from the line's, `matched` shows it, e.g. "pr (MOUZ)", because a namesake on another team is the one
  mis-join names alone can't rule out.
- **Features:** computed as of the line's start time.
- **Unmatched opponents:** a line whose opponent can't be matched gets NO DATA rather than being
  scored against an average team.

The output is sorted by edge:

| Column | Meaning |
|---|---|
| `matched` | The bo3.gg player used. Eyeball it: stand-ins and renamed players are the main mis-join risk |
| `proj` | Projected maps 1-2 total |
| `p_over`, `p_under` | Probabilities |
| `edge` | Probability minus the breakeven implied by the price |
| `pick` | OVER / UNDER / PASS / NO DATA |

## 6. How to know whether it works

The stand-in backtest can't prove an edge against real books, because no historical prop lines exist anywhere
reachable. The real test starts now:

1. Run `lines` at least daily (more often near match times) so `lines_history.csv` builds up.
2. After about 2-3 weeks (several hundred graded lines), grade each archived pick against the bo3.gg
   result and compare the hit rate on picks with `edge > 0` to the breakeven.
3. Only trust the picks if that real hit rate beats the breakeven by more than the sampling noise.
   At 500 picks, ±4.5 points is the 95% band.

Next steps, in order of value:

- **A grading script.** Add it once lines have been archived for about two weeks.
- **A classifier trained on real lines.** Once a few thousand graded lines exist, train it on
  `(features, line) → over?`.
- **A stronger model.** Swap `DecisionTreeRegressor` for `HistGradientBoostingRegressor`. It is a
  one-line change and is usually more accurate.

## 7. Known limitations

- **Maps 1-2 only.** Maps 1-3 needs P(map 3 is played). Thunderpick's `total_maps_over_2_5` price on
  bo3.gg could supply it, but only for some matches.
- **Settlement rules.** The data matches Sleeper's settlement exactly. PrizePicks and Underdog rules on
  overtime, stand-ins and forfeits are unverified; read their help pages.
- **Stand-ins and new rosters.** A player with fewer than 5 maps of history gets NO DATA. A renamed
  player can be fuzzy-matched to the wrong person, so check `matched`.
- **Duplicate player ids.** bo3.gg sometimes splits one person's history across two ids. The most
  recently active id wins.
- **Terms of service.**
  - bo3.gg asks for attribution and marks its robots.txt `ai-train=no`.
  - HLTV bans scraping.
  - Automated use of the DFS apps and PickFinder may breach their terms.
  - Keep request rates low (the code waits about 1 s per request) and use this for personal research.

## Breakeven for pick'em apps

PrizePicks and Underdog pay per entry, not per pick, so `DEFAULT_ODDS` in `model.py` (-119, i.e. 54.3%
per pick, about a 5/6-pick flex entry) sets their breakeven. Change it to match how you play: a 2-pick
power play paying 3x needs about 57.7% per pick (`DEFAULT_ODDS = -137`). Sleeper's per-pick multipliers
and PickFinder's sportsbook prices are used as given.

## Credentials

Never put passwords in code. For `--pickfinder`, set them in the environment:

```bash
export PICKFINDER_EMAIL='you@example.com'
```

```bash
export PICKFINDER_PASSWORD='...'
```

The password in the original script has been shared and should be changed.
