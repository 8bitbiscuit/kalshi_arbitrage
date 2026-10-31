# cs2props cheatsheet

Run everything from the folder that contains `cs2props/`. All files are read from and written to `data/`.

## Setup

```bash
pip install -r cs2props/requirements.txt
```

## Daily routine, in this order

1. Download new match results from bo3.gg. On a first run you can add a start date such as `2025-01-01`.

```bash
python -m cs2props.history
```

2. Fetch current lines from PrizePicks, Underdog and Sleeper. Add `--pickfinder` to include every book
   PickFinder tracks.

```bash
python -m cs2props.lines
```

3. Score the lines. Picks are printed and saved to `data/picks.csv`.

```bash
python -m cs2props.model predict
```

## Model

Retrain about weekly:

```bash
python -m cs2props.model train
```

Compare gradient boosting with a single decision tree:

```bash
python -m cs2props.model compare
```

## Backtesting

Pull PickFinder's past lines and results (needs your login). Add a number of months to change the default of 6.

```bash
python -m cs2props.pickfinder history
```

Grade the model on those lines. Add `data/lines_history.csv` to use your own archive instead, and `tree`
after the file to test the single tree.

```bash
python -m cs2props.model backtest
```

If `history` finds no past lines, record what a player page loads. The responses go to `data/pickfinder_debug/`.

```bash
python -m cs2props.pickfinder discover KaiR0N
```

## PickFinder login

Needed by `lines --pickfinder`, `pickfinder history` and `pickfinder discover`:

```bash
export PICKFINDER_EMAIL='you@example.com'
```

```bash
export PICKFINDER_PASSWORD='...'
```

## Tests

```bash
python -m pytest cs2props -q
```

## Settings

At the top of `cs2props/model.py`:

- `DEFAULT_ODDS` is the breakeven for apps that don't show a price. The default -119 means 54.3% per pick.
  A 2-pick power play needs -137.
- `MODEL` is `"gbm"` for gradient boosting or `"tree"` for a single tree.

## Files in `data/`

| File | Written by | Contents |
|---|---|---|
| `maps.csv` | `history` | One row per player per map from bo3.gg |
| `model_kills.joblib`, `model_headshots.joblib` | `train` | Trained models |
| `lines.csv` | `lines` | Current lines |
| `lines_history.csv` | `lines` | Every scrape, kept for backtesting |
| `picks.csv` | `predict` | Projections, probabilities, edges and picks |
| `prop_history.csv` | `pickfinder history` | PickFinder past lines with results |
| `backtest.csv` | `backtest` | Per-line backtest results |
