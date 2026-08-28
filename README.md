# arbmon

A read-only scanner that watches crypto prediction contracts on **Kalshi** and
**Polymarket**, pairs up contracts that ask the same question, and reports the
cases where buying YES on one venue and NO on the other costs less than the
$1.00 the pair is guaranteed to pay at resolution — after fees.

Nothing in this repository places an order. Neither venue client has an
order-entry method; the collection loop is the whole program.

```
scan #7  kalshi=214 polymarket=63 matched_pairs=28 in 2.4s
[1] CROSS-VENUE  profit $9.96  roi 3.43%  size 300  edge 3.32c/contract
     contract: BTC is above 64999.99 @ 2025-08-27 21:00Z
     buy YES x300    @ 0.4000  fee    $5.04  on kalshi      Bitcoin price at 5pm EDT on Aug 27, 2025?
     buy NO  x300    @ 0.5500  fee    $0.00  on polymarket  Will Bitcoin be above $65,000 on August 27?
     cost $285.00 + fees $5.04 = $290.04   ->  payout $300.00
     ! settlement index differs: Kalshi uses CF Benchmarks, Polymarket uses Binance - the two can disagree at the strike
```

## Is this actually viable?

Short version: the mechanism is real, the edges are real but rare and small,
and most of the difficulty is in *not* finding fake ones. Some things worth
knowing before you point this at real money:

**Kalshi's fee is the dominant term.** It is
`ceil(0.07 × contracts × P × (1−P))`, which peaks at **1.75c per contract** at
a 50c price. A hedged pair costs you that fee on the Kalshi leg alone, so the
two legs must sum to below roughly **98.2c** before you are even at breakeven,
and further below that to be worth the capital. A "1c edge" at midprices is a
loss. `run.py fees` will do this arithmetic for any hypothetical fill.

**Both fees are paid on entry, on the full notional.** You commit ~$0.95 per
contract-pair and get $1.00 back at resolution. The absolute return is small;
the return *rate* depends entirely on how long you wait for settlement, which
is why the report shows ROI on committed capital and why short-dated hourly and
daily contracts are the interesting ones.

**The two venues do not settle on the same number.** Kalshi's crypto markets
settle on CF Benchmarks indices; Polymarket's typically settle on a Binance
candle or a Chainlink feed. Near the strike these disagree. A "riskless" pair
whose legs settle off different indices is not riskless — it is a bet on index
basis, and the scanner labels every such pair rather than pretending otherwise.

**Barrier style is the classic trap.** "Will BTC hit $70k by Friday" (a
one-touch) is worth strictly more than "Will BTC be above $70k at 5pm Friday"
(a terminal digital) with the same strike and expiry. Pairing them looks like
free money and is actually a naked short of the difference. The parser
classifies style explicitly and refuses to cross-match by default.

**Execution risk is not modelled, because the scanner does not execute.** The
two legs are on different venues with separately funded accounts, no
cross-margining, and independent latency. Between spotting an edge and filling
both legs, one side can move and leave you with a naked directional position.
The `slippage_buffer` setting lets you ask "which of these survive a half-cent
of adverse movement per leg", which is a useful reality check.

The honest use of this repo is the one you asked for: **collect the data
first.** Run it for a week, then use `run.py stats` and the SQLite database to
answer whether edges above your fee threshold appear often enough, and survive
long enough, to be worth building execution for.

## Install

```bash
git clone <this repo> && cd kalshi_arbitrage
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10+.

## Quickstart

No credentials needed to start — both venues serve market data publicly:

```bash
python run.py once            # one scan, printed to the terminal
python run.py scan            # loop forever on the configured interval
```

With a config file:

```bash
cp config.example.yaml config.yaml
python run.py -c config.yaml scan
```

### Adding Kalshi credentials (recommended)

Kalshi's `/markets` endpoint is public but only carries top-of-book. The
order book endpoint needs an API key, and without depth the scanner has to
*assume* a size for the Kalshi leg — which means it cannot tell a 5-contract
edge from a 5,000-contract one. Create an API key in Kalshi's account settings,
save the RSA private key, and:

```bash
export KALSHI_API_KEY_ID="your-key-id"
export KALSHI_PRIVATE_KEY_PATH="$HOME/.kalshi/private_key.pem"
```

The client signs each request with RSA-PSS over
`timestamp + METHOD + path`. If the key is missing or rejected, the scanner logs
it once and falls back to top-of-book rather than failing the scan. Polymarket
needs no credentials for reads.

## Commands

| Command | What it does |
|---|---|
| `run.py scan` | Continuous loop. `--interval`, `--max-scans` |
| `run.py once` | A single scan. `--pairs` also prints the match table |
| `run.py markets` | What each venue looks like after parsing. `--venue`, `--limit` |
| `run.py pairs` | Which contracts matched, and (with `--unmatched`) which did not |
| `run.py stats` | Summarise the SQLite collection so far |
| `run.py fees` | Fee for a hypothetical fill: `--venue --price --qty [--bps]` |

`markets` and `pairs` are the tuning tools. If the scanner finds nothing, they
tell you whether it is because nothing parsed, nothing matched, or nothing was
mispriced — three very different problems.

## How it works

```
Kalshi /markets ─┐                                  ┌─ specs_match ──→ "same"      → YES/NO, NO/YES
                 ├─→ normalize → ContractSpec ──────┤
Polymarket ──────┘   (asset, comparator, style,     └─ complementary ─→ "complement" → YES/YES, NO/NO
  /markets + /books    strike, resolve_at, source)
                                                              ↓
                                      walk both ask ladders → size that maximises
                                      profit = size − cost − fees → SQLite + JSONL
```

**Normalisation** (`arbmon/normalize.py`) reduces both venues' phrasings to a
`ContractSpec`. Kalshi's structured `strike_type` / `floor_strike` /
`cap_strike` fields are preferred over its prose; Polymarket has only text, so
it gets a conservative regex parser that returns `None` — dropping the market —
whenever it cannot classify direction and barrier style with confidence. A
missed market costs one trade; a mis-parsed one costs real money.

**Matching** (`arbmon/matching.py`) buckets by asset and resolution window, then
compares specs under a strike tolerance (relative, with an absolute floor for
cheap assets) and a resolution-time window. Every reason for suspicion —
different settlement indices, drifting resolution times, strike-boundary
ambiguity, missing order books — is attached as a caveat that follows the
opportunity into the report and the database.

**Order book handling** (`arbmon/clients/`) is where the subtle bug lives, so it
is isolated and tested. Kalshi does not publish an ask ladder: its book is two
lists of resting *bids*, and a resting NO bid at 36c *is* an offer to sell YES
at 64c. `derive_ask_ladders` does that inversion. Polymarket's YES and NO are
separate tokens with separate books, both of which are fetched, so buying NO is
a real purchase rather than a synthetic short.

**Sizing** (`arbmon/arbitrage.py`) walks both ask ladders together rather than
comparing best asks, because the top of book might be five contracts deep. The
profit-maximising size always lies on a cumulative depth boundary — between
boundaries the marginal cost of another contract is constant, so you either
want the whole segment or none of it — which makes evaluating the boundaries
exact rather than approximate.

**Fees** (`arbmon/fees.py`) implement Kalshi's `ceil(0.07 × C × P × (1−P))`
(0.035 for the index series) and Polymarket's
`rate × min(P, 1−P) × C`, reading Polymarket's per-market rate from the API
where it reports one. Optional flat per-leg costs let you charge amortised
transfer costs against each trade.

## Output

Opportunities go to the terminal, to `data/opportunities.jsonl`, and to SQLite
at `data/arbmon.sqlite`:

- **`scans`** — one row per scan: counts, timing, and any venue error.
- **`snapshots`** — every quote seen, with the depth ladders as JSON. This is
  the table that answers "how long do edges survive?".
- **`opportunities`** — every reported opportunity with its legs, fees and
  caveats. `signature` is stable across scans, so repeat sightings of the same
  edge group together.

Snapshot recording adds roughly one row per market per scan; turn it off with
`storage.record_snapshots: false` if the file grows faster than you want.

## Tuning

The settings that most change what you see:

- `arbitrage.min_profit` / `min_edge_per_contract` / `min_roi` — an opportunity
  must clear all three. Defaults are deliberately conservative.
- `arbitrage.slippage_buffer` — added to every fill price. Set it to `0.005` to
  see which edges survive half a cent of adverse movement per leg.
- `arbitrage.require_depth` — only report opportunities backed by a real order
  book on both legs. Worth turning on once you have Kalshi credentials.
- `matching.strike_rel_tolerance` — tighten if you see adjacent strikes pairing
  up; loosen if venues quote the same strike slightly differently.
- `matching.allow_style_mismatch` — leave `false`. See above.

## Tests

```bash
python -m pytest
```

179 tests, no network access required. The suite covers the fee formulas
against Kalshi's published schedule, the order book inversion, the parsers'
refusal to guess, cross-venue matching and its caveats, depth-aware sizing
(including the cases where the second level destroys the edge), and a full
end-to-end scan driven through mocked HTTP transports with realistic payloads.

## Layout

```
run.py                    CLI
arbmon/
  config.py               dataclass config + YAML loading
  models.py               ContractSpec, NormalizedMarket, Leg, Opportunity
  fees.py                 Kalshi and Polymarket fee schedules
  normalize.py            venue text/fields -> ContractSpec
  matching.py             cross-venue pairing + caveats
  arbitrage.py            ladder sweeping, sizing, detection
  scanner.py              orchestration loop
  storage.py              SQLite + JSONL
  report.py               console output
  clients/
    base.py               async HTTP, rate limiting, retries
    kalshi.py             trade-api v2, RSA-PSS signing, book inversion
    polymarket.py         Gamma metadata + CLOB books
```

## Notes

Both venues have terms of service governing API use, automated trading, and
eligibility by jurisdiction; read them before doing anything with this beyond
collecting data. Rate limits are configurable per venue and default to a polite
8 requests/second.
