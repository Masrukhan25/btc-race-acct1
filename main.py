"""RegimeDip v1d: a long-only CBBTC dip/rebound bot for the 24h BTC race.

The bot watches one-minute cbBTC bars and only ever buys the whole cash
balance. It looks for two kinds of washed-out long: a shallow dip inside an
otherwise healthy trend (RSI(14) under 45 with a 60-bar washout), and a
deeper rebound off a long fall. Every trade carries a stop and either a
take-profit or an armed trailing stop, and the bot never holds more than one
position, never enters more than four times per window, and rests sixty bars
after every exit.

Wire protocol is agent-fun/bars-replay/v1, spoken through the vendored shim
in vendor/makefun_bars.py. All prices are integers; every percentage is
basis points against UNIT = 10000. No clock, no network, no randomness: the
same bars always give the same orders.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor"))

import makefun_bars

SYM = "CBBTC"
UNIT = 10000
MAX_SLIP = 200
START_BARS = 62          # bars of history before the first entry is allowed
LAST_ENTRY_SEQ = 1320    # no new entries after this batch seq
MAX_ENTRIES = 4          # at most four entries per window
REST_BARS = 60           # bars of rest after an exit
HOLD_BARS = 300          # forced exit after this many bars in a trade
EQUITY_FLOOR_BPS = 9800  # stop entering below 98% of starting capital
# Dip setup (shallow pullback inside a healthy trend)
DIP_TOP_DROP_MIN = 100   # fall from the 60-bar high to the previous close
DIP_FALL720_MAX = 550    # the 720-bar fall must stay under this
DIP_RSI_MAX = 45         # RSI(14) must be under this
DIP_FALL60_MIN = 80      # 60-bar close-to-close fall, at least this
DIP_REGIME_MIN = -100    # 360-bar return must be at least this (bps)
DIP_STOP_BPS = 60        # stop below the entry close
DIP_TAKE_BPS = 120       # take-profit above the entry close
DIP_MAX = 2              # at most this many dip trades per window
# Rebound setup (deep fall, wick + reversal)
RB_FALL60_MIN = 150
RB_FALL120_MIN = 250
RB_FALL720_MIN = 550
RB_WICK_MIN = 20         # wick bar range in bps
RB_WICK_TOP = 60         # close must sit in the top 40% of its bar
RB_REV_MIN = 60          # reversal strength in bps
RB_STOP_LOW = 100        # stop at most this far below the entry close
RB_STOP_HIGH = 50        # stop at least this far below the entry close
RB_STOP_PAD = 20         # stop a little under the 720-bar low
RB_ARM = 80              # trailing stop arms this far above entry
RB_TRAIL = 80            # trailing stop distance once armed
RB_MAX = 2               # at most this many rebound trades per window


class RegimeDip:
    """State and decisions for one window."""

    def __init__(self):
        self.cash0 = 0
        self.closes = []
        self.highs = []
        self.lows = []
        self.rsi = []          # Wilder RSI(14), aligned with closes, None pre-seed
        self.avg_gain = 0.0
        self.avg_loss = 0.0
        self.seen_t = set()
        self.pending = None    # setup waiting for its fill: kind/stop/take/arm/trail/hold
        self.trade = None      # open trade: kind/entry/stop/take/arm/trail/hold/held/peak/armed/exit_sent
        self.rest_until = -1
        self.entries = 0
        self.dip_n = 0
        self.rb_n = 0
        self.rb_loss_streak = 0

    # -- history ---------------------------------------------------------

    def add_bar(self, bar):
        """Fold one bar into the series and refresh RSI(14)."""
        close = int(bar["c"])
        self.closes.append(close)
        self.highs.append(int(bar["h"]))
        self.lows.append(int(bar["l"]))
        n = len(self.closes) - 1
        self.rsi.append(None)
        if n == 0:
            return
        change = close - self.closes[n - 1]
        gain = change if change > 0 else 0
        loss = -change if change < 0 else 0
        if n < 14:
            return
        if n == 14:
            ups = 0
            downs = 0
            for i in range(1, 15):
                step = self.closes[i] - self.closes[i - 1]
                if step > 0:
                    ups += step
                else:
                    downs += -step
            self.avg_gain = ups / 14.0
            self.avg_loss = downs / 14.0
        else:
            self.avg_gain = (self.avg_gain * 13 + gain) / 14.0
            self.avg_loss = (self.avg_loss * 13 + loss) / 14.0
        if self.avg_loss == 0:
            self.rsi[n] = 100.0
        else:
            strength = self.avg_gain / self.avg_loss
            self.rsi[n] = 100.0 - 100.0 / (1.0 + strength)

    # -- measurement helpers --------------------------------------------

    def fall_from_top(self, span):
        """Fall in bps from the highest high of the last `span` bars to the
        previous close. Returns 0 when there is not enough history."""
        if len(self.highs) < span + 1:
            return 0
        top = max(self.highs[-span - 1:-1])
        prev = self.closes[-2]
        return (top - prev) * UNIT // top

    def close_fall(self, span):
        """Close-to-close fall in bps over `span` bars, ending at the previous
        close. Returns 0 when there is not enough history."""
        if len(self.closes) < span + 2:
            return 0
        return (self.closes[-span - 2] - self.closes[-2]) * UNIT // self.closes[-span - 2]

    def has_wick(self):
        """True when some bar in the last 20 has a wide range (>= 20 bps) and
        closes in the top 40% of its own bar."""
        start = max(0, len(self.closes) - 20)
        for i in range(start, len(self.closes)):
            hi = self.highs[i]
            lo = self.lows[i]
            cl = self.closes[i]
            if hi <= lo:
                continue
            if (hi - lo) * UNIT // cl < RB_WICK_MIN:
                continue
            if (cl - lo) * 100 // (hi - lo) >= RB_WICK_TOP:
                return True
        return False

    # -- setups ----------------------------------------------------------

    def dip_setup(self):
        """Shallow dip inside a healthy trend. Returns the pending-order plan
        or None."""
        cl = self.closes
        prev = cl[-2]
        cur = cl[-1]
        top60 = max(self.highs[-61:-1])
        if (top60 - prev) * UNIT // top60 < DIP_TOP_DROP_MIN:
            return None
        if self.fall_from_top(720) >= DIP_FALL720_MAX:
            return None
        r1 = self.rsi[-1]
        r2 = self.rsi[-2]
        if r1 is None or r2 is None or r1 >= DIP_RSI_MAX:
            return None
        if cur <= prev:
            return None
        if self.close_fall(60) < DIP_FALL60_MIN:
            return None
        if len(cl) >= 361 and (cl[-1] - cl[-361]) * UNIT // cl[-361] < DIP_REGIME_MIN:
            return None
        return {
            "kind": "dip",
            "stop": cur * (UNIT - DIP_STOP_BPS) // UNIT,
            "take": cur * (UNIT + DIP_TAKE_BPS) // UNIT,
            "arm": 0,
            "trail": 0,
            "hold": HOLD_BARS,
        }

    def rebound_setup(self):
        """Deep rebound off a long fall. Returns the pending-order plan or
        None."""
        cl = self.closes
        prev = cl[-2]
        cur = cl[-1]
        leg60 = self.close_fall(60) >= RB_FALL60_MIN
        leg120 = self.close_fall(120) >= RB_FALL120_MIN
        if not (leg60 or leg120):
            return None
        if self.fall_from_top(720) < RB_FALL720_MIN:
            return None
        if not self.has_wick():
            return None
        r1 = self.rsi[-1]
        r2 = self.rsi[-2]
        if r1 is None or r2 is None:
            return None
        if (cur - prev) * UNIT // prev < RB_REV_MIN:
            return None
        if cur <= self.highs[-2]:
            return None
        if not r1 > r2:
            return None
        top720 = max(self.highs[-721:-1])
        low720 = min(self.lows[-721:-1])
        stop0 = low720 * (UNIT - RB_STOP_PAD) // UNIT
        lo_stop = cur * (UNIT - RB_STOP_LOW) // UNIT
        hi_stop = cur * (UNIT - RB_STOP_HIGH) // UNIT
        stop = min(max(stop0, lo_stop), hi_stop)
        if top720 - cur < 2 * (cur - stop):
            return None
        return {
            "kind": "rb",
            "stop": stop,
            "take": None,
            "arm": RB_ARM,
            "trail": RB_TRAIL,
            "hold": HOLD_BARS,
        }

    # -- orders ----------------------------------------------------------

    def buy_order(self, state):
        return {
            "side": "buy",
            "symbol": SYM,
            "notional_quote_units": state.cash,
            "max_slippage_bps": MAX_SLIP,
        }

    def sell_order(self, pos):
        return {
            "side": "sell",
            "symbol": SYM,
            "qty": pos.qty,
            "max_slippage_bps": MAX_SLIP,
        }

    # -- trade management -------------------------------------------------

    def manage_trade(self, state, seq):
        """Run one batch of exit management. Returns a list of orders, []
        when holding without an exit, or None when the trade is gone (no
        position) so the caller can look for a new entry."""
        trade = self.trade
        pos = state.positions.get(SYM)
        if pos is None:
            self.trade = None
            return None
        if trade["exit_sent"]:
            return [self.sell_order(pos)]
        close = self.closes[-1]
        trade["held"] += 1
        if close > trade["peak"]:
            trade["peak"] = close
        if not trade["armed"] and close >= trade["entry"] * (UNIT + trade["arm"]) // UNIT:
            trade["armed"] = True
        exit_px = None
        if trade["kind"] == "dip":
            if close >= trade["take"]:
                exit_px = close
            elif close <= trade["stop"]:
                exit_px = trade["stop"]
            elif trade["held"] >= trade["hold"]:
                exit_px = close
        else:
            if close <= trade["stop"]:
                exit_px = trade["stop"]
            elif trade["armed"] and close <= trade["peak"] * (UNIT - trade["trail"]) // UNIT:
                exit_px = trade["peak"] * (UNIT - trade["trail"]) // UNIT
            elif trade["held"] >= trade["hold"]:
                exit_px = close
        if exit_px is None:
            return []
        if trade["kind"] == "rb":
            if exit_px > trade["entry"]:
                self.rb_loss_streak = 0
            else:
                self.rb_loss_streak += 1
        self.rest_until = seq + REST_BARS
        trade["exit_sent"] = True
        return [self.sell_order(pos)]

    # -- main decision ----------------------------------------------------

    def on_batch(self, state, batch):
        bar = state.latest.get(SYM)
        if bar is None:
            return []
        stamp = batch.get("t")
        if stamp in self.seen_t:
            return []
        self.seen_t.add(stamp)
        self.add_bar(bar)
        seq = state.seq

        if self.pending is not None:
            plan = self.pending
            self.pending = None
            pos = state.positions.get(SYM)
            if pos is not None:
                entry = int(bar["o"])
                self.trade = {
                    "kind": plan["kind"],
                    "entry": entry,
                    "stop": plan["stop"],
                    "take": plan["take"],
                    "arm": plan["arm"],
                    "trail": plan["trail"],
                    "hold": plan["hold"],
                    "held": 0,
                    "peak": entry,
                    "armed": False,
                    "exit_sent": False,
                }
                self.entries += 1
                if plan["kind"] == "dip":
                    self.dip_n += 1
                else:
                    self.rb_n += 1

        if self.trade is not None:
            orders = self.manage_trade(state, seq)
            if orders is not None:
                return orders

        if state.warming_up:
            return []
        if len(self.closes) < START_BARS:
            return []
        if seq > LAST_ENTRY_SEQ:
            return []
        if self.entries >= MAX_ENTRIES:
            return []
        if seq < self.rest_until:
            return []
        if state.equity < self.cash0 * EQUITY_FLOOR_BPS // UNIT:
            return []

        if self.rb_n < RB_MAX and self.rb_loss_streak < 2:
            plan = self.rebound_setup()
            if plan is not None:
                self.pending = plan
                return [self.buy_order(state)]
        if self.dip_n < DIP_MAX:
            plan = self.dip_setup()
            if plan is not None:
                self.pending = plan
                return [self.buy_order(state)]
        return []


BOT = RegimeDip()


def on_hello(state):
    BOT.cash0 = int(state.hello["capital"]["quote_units"])


def on_batch(state, batch):
    return BOT.on_batch(state, batch)


if __name__ == "__main__":
    makefun_bars.run(on_batch, on_hello=on_hello)
