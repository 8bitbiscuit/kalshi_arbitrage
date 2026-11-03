# CS2 player-prop model: plan and how to run it

Goal: for each current "Maps 1-2 Kills" / "Maps 1-2 Headshots" line (PrizePicks, Underdog, Sleeper,
optionally every book PickFinder tracks), predict whether the player goes over or under, using a
gradient-boosted decision trees trained on CS2 match history.

```
bo3.gg API ──history.py──▶ data/maps.csv ──model.py train──▶ data/model_*.joblib
                                                              │
DFS / PickFinder ──lines.py──▶ data/lines.csv ──model.py predict──▶ data/picks.csv
                         └──▶ data/lines_history.csv ─┐
PickFinder history ──pickfinder.py──▶ data/prop_history.csv ─┴─model.py backtest──▶ data/backtest.csv
```

## Run it

```bash
pip install -r cs2props/requirements.txt
python -m cs2props.history          # first run about 1 hour (all of CS2); later runs take a minute
python -m cs2props.model train      # prints honest test metrics, saves the two models
python -m cs2props.model compare    # gradient boosting vs a single decision tree on the same test data
python -m cs2props.lines            # current lines from PrizePicks, Underdog, Sleeper (no login)
python -m cs2props.model predict    # prints and saves picks, best edge first
python -m cs2props.pickfinder history            # past lines + results from PickFinder (your login)
python -m cs2props.model backtest                # grade the model on them (or: backtest data/lines_history.csv)
python -m pytest cs2props -q        # self-checks (leakage, probabilities, name matching, end to end)
```

Daily: `history` then `lines` then `predict`. Retrain weekly or after big roster moves.

After updating the package: run `train` again (saved models remember their feature list and `predict`
refuses old ones), and re-run `pickfinder history` so past lines get their market win probabilities.
Then `backtest` learns the calibrator that `predict` uses.

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
| `map_kpr`, `map_factor` | Likely maps: how often **both teams** picked each map recently (maps 1-2 of their best-of-3s among their last 30 maps), and the player's KPR on each (shrunk toward their overall KPR with a 100-round prior). `map_kpr` is that weighted KPR; `map_factor` is it divided by their overall KPR | Map Picks + per-map Kills/KPR |
| `win_prob`, `elo_opp` | Per-map win probability for the player's team: the market's, when the line came with one (PickFinder), converted from series to map odds; otherwise from our map-level Elo. A team id seen for the first time (a roster moving org, e.g. LP to BARCA) starts from its players' previous teams' ratings | Moneyline implied win % |
| (inside `exp_kills_m12`) | How even the match should be, as expected rounds per map from a round-by-round model of first-to-13 with overtime: about 23 rounds for an even map, 21 when one side is a 90% favourite. As separate inputs, closeness and expected rounds made projections jump by kills on a 0.01 change in win probability (the trees split on noise), with no accuracy gain, so they only enter through the expected-kills feature | Potential Blowout, match score odds |
| `opp_rounds` | How long the opponent's recent maps ran (the player's own team is in `rounds_l20`) | |
| `exp_kills_m12`, `exp_hs_m12` | Expected maps 1-2 kills: 2 × expected rounds (from closeness) × KPR on likely maps (× HS% for headshots). The model's most important input | |
| `maps_played` | Experience; lets the model distrust thin histories | sample size |

Why map pool, not actual maps: lines lock before the veto, and bo3.gg has had no veto data since
mid-September 2026. Using the actual maps in training but not at prediction time would teach the model
something it can't use live.

Where the market win probability comes from: PickFinder records both teams' implied win chances with every
past line (Prop History) and with every current prop. `pickfinder history` stores it with each past line.
`lines --pickfinder` writes one odds-only row per match (stat `match_odds`), so matches PickFinder lists
only for PrizePicks, Underdog or Sleeper still get it. Any line without it (PrizePicks, Underdog and Sleeper's own
feeds) borrows it from another line of the same match, else falls back to our Elo. The training data
(90,000 bo3.gg series) has no odds, so the model learns from Elo; the market only replaces Elo when lines
are scored. Both describe the same thing, the chance the player's team wins a map.

Likely maps come from maps 1-2 of best-of-3s because those are always one team's pick (map 3 is the
decider). bo3.gg's own pick/ban records stopped in mid-September 2026.

## 3. Model

- **Target:** maps 1-2 kills (and a second model for headshots), for every player in every BO3/BO5 that
  reached map 2. A *regression* model, because there are no historical lines to label over/under.
- **Model:** gradient boosting (`HistGradientBoostingRegressor`). It is hundreds of small decision trees,
  each fitted to the errors the earlier ones still make, added up in small steps. Early stopping ends
  training once new trees stop helping. A single decision tree is still available (`MODEL = "tree"`, or
  `backtest <csv> tree`) for comparison.
- **Over/under probability:** each validation result gives a held-out error `z = (actual − predicted) / √predicted`.
  The z values are centred so a level shift between periods doesn't bias every pick. P(over 30.5) is
  the share of z values for which `projection + z·√projection` lands above 30.5. Counting stats spread
  like √mean, so one z distribution fits 15-kill and 35-kill players alike. Whole-number lines
  (PrizePicks posts 35, 27, ...) get push handling.
- **Tuning:** loss (squared error or Poisson, which suits counts), learning rate 0.05 or 0.1, and 15 or
  31 leaves per tree, chosen on a validation slice by MAE. Both stats chose Poisson, 0.05 and 15.
- **Split by time:** oldest 70% train, next 15% validate (tuning and the error spread), newest 15% test.
- **Recent matches count more:** each training match is weighted by 0.5 per 90 days before the newest one
  (`HALF_LIFE_DAYS`). On bo3.gg data this removed the headshot projections' slight low bias (0.03 → 0.00)
  and only nudged kills (0.24 → 0.21 too low), at a small accuracy cost (6.047 → 6.057 kills error).
- **Kills corrected by their recent miss:** against real lines the kills projections ran low, mostly for
  the standout players books post props on, whom the trees pull toward the average. Each kills projection
  gets the average (actual − projection) of real kills lines from the last 30 days added (`MISS_DAYS`),
  once at least 50 player-series exist (`MISS_MIN`). The backtest measures this walk-forward: each line
  only uses results from matches at least a day before it. It then saves the latest 30 days' figure to
  `data/kills_miss.json` for `predict`. Every backtest rewrites that file (0 when there are too few series)
  and the calibrator records the same figure, so `predict` never pairs a calibrator with a correction from
  another run. Headshots are not corrected, because the recency weights already remove their bias.
- **Calibrated on real lines:** the model alone was far too confident against real books (see section 6),
  so `backtest` learns a small logistic regression from real graded lines: how often a line went over,
  given how far the projection was from it (per stat). It is fitted on matches before a cut-off (about 60%
  of lines) and judged on the matches after it. It is saved to `data/calibrator.joblib` only if it beats
  the raw model's log loss on those later matches, and only from at least 300 player-series in 30 matches
  (copies of a line at several books count once). `predict` uses it only if it was learned with the same
  model and features, and says how many series it came from. The model still supplies the push chance.
- **Picks ignore pushes:** a push is refunded, so a pick compares P(over | no push) with the breakeven.
- **What `train` prints:** test MAE of the model versus the naive "2 × 20-map average", a stand-in-line
  hit rate (optimistic, since real books are sharper than a 20-map average), Brier score versus a coin
  flip, and the top features (how much the test error rises when each feature is shuffled).

### Results on real data

Trained on bo3.gg data from July 2025 to October 2026, about 90,000 player-series. Tested on the newest
15% (2026-09-16 to 2026-10-30), which the model never saw:

| | Kills, maps 1-2 | Headshots, maps 1-2 |
|---|---|---|
| **Gradient boosting**, mean absolute error | **6.03** | **3.83** |
| Single decision tree | 6.08 | 3.87 |
| Naive "2 × 20-map average" | 6.33 | 3.97 |
| Gain over the tree, paired 95% band | 0.054 ± 0.018 | 0.035 ± 0.011 |
| Brier on the stand-in line, boosting / tree (0.250 = coin flip) | 0.235 / 0.237 | 0.238 / 0.241 |
| Distinct projections on the 13,618 test rows, boosting / tree | 6,768 / 54 | 7,184 / 56 |
| Most important features | kills per round, map factor, win probability, opponent Elo | 20-map headshots, HS% |

Adding the likely-maps, closeness and expected-rounds features left these errors unchanged on bo3.gg
data (6.057 → 6.051 kills, 3.838 → 3.834 headshots; both gains are inside their paired 95% bands of 0.012
and 0.006). The new expected-kills feature became the model's top input, but it repackages what the model
could already combine from KPR, Elo and map form. Its value should show on real lines, where the market's
win probability is sharper than Elo, which only the backtest on your PickFinder lines can measure.

Boosting beats the single tree on both stats by a small margin that the paired test says is real. It
also gives every player their own projection: the tree put everyone into about 55 groups that shared one
number each.

Backtest on stand-in lines (6,802 lines on real September-October series, trained on data before
September), boosting vs tree:

| | Gradient boosting | Single tree |
|---|---|---|
| Hit rate (±2.3 points) | 61.4% | 60.7% |
| ROI | 14.2% | 12.8% |
| Hit rate on bets with 10%+ edge | 71.3% | 67.0% |

The stand-in lines are softer than real books, so only the comparison means anything, and the overall
gap is inside the noise. Both models also predicted P(over) about 5 points too low in that period,
because kill levels rose after their training cutoff. Real PickFinder lines are the deciding test.

On the 2026-10-30 slate (156 lines), projections averaged within 0.03 kills and 0.21 headshots of the
books' lines (standard deviations 1.5 and 1.4). The model agrees with the books on level, so any edge
lives in disagreements of 1-2 kills. That is also why the stand-in hit rate (about 60%) is no guide to
real profit.

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

## 6. How to know whether it works: backtest on real past lines

The stand-in check in `train` can't prove an edge against real books. Only real past lines can, and there
are two sources:

| Source | Command | What it gives |
|---|---|---|
| PickFinder prop history (about 6 months, main lines only) | `python -m cs2props.pickfinder history` | `data/prop_history.csv`: past lines and results for every player with a prop up today. Run it on several days to cover more players |
| Our own archive | every `lines` run appends to `data/lines_history.csv` | Exact closing lines and prices from PrizePicks, Underdog and Sleeper, from now on |

`python -m cs2props.model backtest [csv]` then grades the model on those lines:

- **Matching:** each line is first matched to its own bo3.gg series: same player, same opponent, starting
  within 90 minutes. Book and bo3.gg times differ by minutes, while same-day rematches are 2+ hours apart.
  Lines whose series isn't in bo3.gg are not graded rather than graded against the wrong match.
- **No look-ahead:**
  - The trees are refit on matches *before* the first graded line.
  - Features are taken strictly before the *earlier* of the book's and bo3.gg's start time. Books and
    PickFinder often list a start a few minutes after bo3.gg does, and using the later time let the graded
    match into its own features (a test now checks this).
- **Closing line:** for our archive, each book's last snapshot before that time, one per series.
- **Results** come from the file (what settled) and are cross-checked against bo3.gg. The agreement rate
  is printed.
- **The report** covers:
  - hit rate with its 95% band, versus breakeven. The band counts distinct player-series, because the
    same line at three books is one outcome, not three
  - units won and ROI at each line's own price
  - all of that split by edge bucket, stat, OVER/UNDER and book
  - "always over / always under" baselines
  - a calibration table (predicted P(over) vs actual)
  - the share of lines that had a market win probability
- **Kills correction:** it prints kills bets before and after the average-miss correction, including the
  share of OVER picks, and saves the latest correction for `predict`. On stand-in lines the OVER share of
  kills picks went from 33% to 48%.
- **Calibrator:** with 300+ player-series from 30+ matches it then fits the calibrator on earlier matches,
  prints raw vs calibrated results on later ones (overall, by edge, calibration table, log loss), and
  saves it for `predict` only if its log loss is lower. Otherwise the existing file is left alone. A
  `tree` backtest never saves one. On a one-day archive of 307 lines (154 series) the first version saved a
  calibrator that cut the later hit rate from 51.6% to 39.6%; these rules stop that. The calibrator is fitted
  on corrected kills projections, so `predict` ignores one saved before the correction existed.
- **Per-line results** go to `data/backtest.csv`.

Only trust the picks if the hit rate beats breakeven by more than the band. At 500 bets the band is
about ±4.5 points, so a 58% hit rate at a 54.3% breakeven is not yet proof.

**How `pickfinder history` gets past lines.** A logged-in `discover` run on 2026-11-02 found the request
behind the Prop History panel. It is a website route, keyed by the player's OpticOdds id:

```
https://www.pickfinder.app/api/players/{player_oo_id}/props/historical?stat=Maps%201-2%20Kills
```

It returns one entry per past match with the line, each book's prices and both teams' implied win
chances, but no result. Its match key is the scheduled US Eastern date plus the opponent's id, which is
the same id the player's game log (`/players/{id}/gamelog`) uses. So each entry is joined to the series
against that opponent within a day of that date, and the result is the sum of maps 1 and 2 from the
game log:

- **Skipped, not guessed:** entries with no such series (not played yet, or the player sat out a map) or
  two equally close ones.
- **Each book's own line:** a book's line can differ from the entry's. PrizePicks' is read from its link,
  e.g. 28.5 where the entry says Underdog's 26.5.
- **Main lines only:** goblin, demon and other one-sided lines are dropped.

Checked on huNter's real responses: 20 of his 22 entries matched a series. The other two were an
unplayed match and one missing from his game log. The backtest then matched all 21 resulting lines to
bo3.gg, and PickFinder's results agreed with bo3.gg on all 21. The scraper's login and paging around
these calls are still untested end to end, because they need your login.

PickFinder's prices for pick'em apps are its per-pick equivalents: -119 for PrizePicks, -107 to -115 for
Underdog. The backtest uses them as given.

`discover` stays available for when PickFinder changes its site. It records every PickFinder data request
a player page makes, with responses, while you click around, and saves them to `data/pickfinder_debug/`.

Scraping many player histories is heavier use than the app's own pages make. The code waits 1 s per
request, but check PickFinder's terms before running it often.

Next steps, in order of value:

- **More inputs to the calibrator** once more lines exist: the market's closeness, the book, how far
  the line moved.
- **Monthly walk-forward refits in the backtest.** It trains once, before the first graded line. A check
  on bo3.gg data found this costs little: a model trained before August projected only about 0.2 kills
  too low by October, and monthly refits barely changed that. Refits would also make the saved kills
  correction exact: it is measured against the backtest's older models, so `predict`'s fresher models may
  be corrected slightly too much.

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
