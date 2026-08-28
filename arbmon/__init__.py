"""arbmon - cross-venue arbitrage scanner for crypto prediction markets.

Collects order book data from Kalshi and Polymarket, normalises the contracts
into a common representation, pairs up economically equivalent contracts and
reports pairs whose combined cost is below the $1.00 settlement payout after
fees.

This package only *reads* market data. Nothing in here places orders.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
