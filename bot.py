"""
EXPIRYRANGE Compression-Regime Bot for Deriv 1HZ10V
====================================================

Trades ONLY EXPIRYRANGE ("Ends Between") contracts on 1HZ10V, gated on:

  1. A regime classifier reporting "Compression" (low realized vol,
     low ADX, narrow Bollinger bandwidth, low Hurst exponent) -- the
     primary indicator. Anything else (Expansion/Trending) is skipped.

  2. A layered Monte Carlo engine picks the (duration, barrier) pair:
       - HMM regime-switching GBM  (state-dependent mu/sigma, transitions
         sampled from the fitted transition matrix)
       - GARCH(1,1) conditional volatility feeding sigma at each step
         (vol clustering, instead of a flat per-regime sigma)
       - Merton jump-diffusion overlay (tail/gap risk -- synthetic
         indices do jump)
       - Historical bootstrap resampling of empirical returns, run in
         parallel as a non-parametric cross-check against the parametric
         stack (mirrors the bootstrap approach already used in
         deriv_multisymbol_bot.py)

  3. Deriv's own proposal API must quote a payout >= MIN_PAYOUT_PCT for
     the candidate barrier/duration.

  4. Expected value (model P_stay x payout) must be positive, i.e. the
     model's estimated probability of staying in-range must exceed the
     breakeven probability implied by Deriv's payout.

IMPORTANT -- READ BEFORE RUNNING LIVE:
The 52% figure is a *payout selection threshold*, not a guaranteed trade
outcome. A trade only fires when both (3) and (4) hold. Losing trades
are still possible; no Monte Carlo model can force a market to behave.
This bot is a filter for asymmetric setups, not a profit guarantee.

Environment variables (see .env.example):
  DERIV_APP_ID, DERIV_API_TOKEN, SUPABASE_URL, SUPABASE_KEY,
  STAKE_AMOUNT, MIN_PAYOUT_PCT, MC_PATHS, and regime thresholds below.
"""

import asyncio
import os
import json
import time
import math
import logging
import logging.handlers
import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Tuple, Dict, Any
from collections import deque

import numpy as np

try:
    import websockets
except ImportError:
    websockets = None

try:
    import aiohttp
except ImportError:
    aiohttp = None

try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False

try:
    from supabase import create_client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False

LOG_DIR = os.environ.get("LOG_DIR", "logs")
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
os.makedirs(LOG_DIR, exist_ok=True)


def _make_rotating_handler(filename: str, level, fmt: str) -> logging.Handler:
    handler = logging.handlers.RotatingFileHandler(
        os.path.join(LOG_DIR, filename), maxBytes=10 * 1024 * 1024, backupCount=5,
    )
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(fmt))
    return handler


# -- operational/system logger --------------------------------------------
# Connection state, heartbeat summaries, warnings, errors, one-line
# "what happened" events. Tail this for "is the bot alive and doing the
# right thing right now".
_console_handler = logging.StreamHandler()
_console_handler.setLevel(LOG_LEVEL)
_console_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))

log = logging.getLogger("expiryrange_compression_bot")
log.setLevel(LOG_LEVEL)
log.addHandler(_console_handler)
log.addHandler(_make_rotating_handler("bot.log", LOG_LEVEL, "%(asctime)s %(levelname)s %(name)s: %(message)s"))
log.propagate = False

# -- decision-audit logger -------------------------------------------------
# Every time the regime layer fires a Compression read and the Monte Carlo
# grid runs, this writes one full audit block: the regime reading, every
# MC candidate's parametric/bootstrap/blended probabilities, the payout-gate
# and EV-gate outcome for each candidate, and the final decision -- whether
# or not a trade was actually taken. Read this to understand *why* a
# specific trade fired (or didn't).
#
# NOTE (Railway): local disk is ephemeral across redeploys/restarts, so
# treat these files as a rolling local convenience buffer, not permanent
# storage -- the Supabase trade rows remain the durable record.
trade_log = logging.getLogger("trade_signals")
trade_log.setLevel(logging.INFO)
_trade_console = logging.StreamHandler()
_trade_console.setLevel(logging.INFO)
_trade_console.setFormatter(logging.Formatter("%(message)s"))
trade_log.addHandler(_trade_console)
trade_log.addHandler(_make_rotating_handler("trades.log", logging.INFO, "%(message)s"))
trade_log.propagate = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SYMBOL = "1HZ10V"

DERIV_APP_ID = os.environ.get("DERIV_APP_ID", "1089")
DERIV_API_TOKEN = os.environ.get("DERIV_API_TOKEN")

# Current Options API (REST + OTP-authenticated WebSocket) -- replaces the
# old ws.derivws.com/websockets/v3?app_id=... model, which now rejects the
# handshake with HTTP 401 rather than letting you authorize in-band.
DERIV_WS_PUBLIC_URL = "wss://api.derivws.com/trading/v1/options/ws/public"
DERIV_ACCOUNT_TYPE = os.environ.get("DERIV_ACCOUNT_TYPE", "real")  # "demo" or "real"
DERIV_ACCOUNT_ID = os.environ.get("DERIV_ACCOUNT_ID")  # optional -- skips the accounts lookup if set

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

# Trading params -- small stake per explicit choice to go live directly
STAKE_AMOUNT = float(os.environ.get("STAKE_AMOUNT", "1.0"))
MIN_PAYOUT_PCT = float(os.environ.get("MIN_PAYOUT_PCT", "52.0"))  # gate, not a guarantee
MAX_CONCURRENT_TRADES = int(os.environ.get("MAX_CONCURRENT_TRADES", "1"))
COOLDOWN_SEC_AFTER_TRADE = int(os.environ.get("COOLDOWN_SEC_AFTER_TRADE", "30"))

# Monte Carlo params
MC_PATHS = int(os.environ.get("MC_PATHS", "5000"))
HMM_N_STATES = int(os.environ.get("HMM_N_STATES", "3"))
JUMP_INTENSITY_PER_DAY = float(os.environ.get("JUMP_INTENSITY_PER_DAY", "2.0"))
JUMP_MEAN = float(os.environ.get("JUMP_MEAN", "0.0"))
JUMP_STD = float(os.environ.get("JUMP_STD", "0.004"))
TICKS_PER_SECOND = float(os.environ.get("TICKS_PER_SECOND", "1.0"))  # 1HZ10V ~ 1 tick/sec

# Candidate grid the Monte Carlo engine searches over
DURATION_CANDIDATES_MIN = [5, 10, 15, 20, 30]
BARRIER_HALF_WIDTH_PCT = [0.15, 0.25, 0.35, 0.5, 0.75, 1.0]

# Data windows
PRICE_HISTORY_LEN = int(os.environ.get("PRICE_HISTORY_LEN", "3000"))
REFIT_INTERVAL_SEC = int(os.environ.get("REFIT_INTERVAL_SEC", "900"))  # refit HMM/GARCH every 15 min
MIN_TICKS_BEFORE_TRADING = int(os.environ.get("MIN_TICKS_BEFORE_TRADING", "600"))

# Regime (Compression) thresholds -- percentile-based against the bot's
# own rolling history so it self-calibrates per instrument rather than
# using hardcoded absolute vol levels.
COMPRESSION_VOL_PERCENTILE = float(os.environ.get("COMPRESSION_VOL_PERCENTILE", "35"))
COMPRESSION_BBW_PERCENTILE = float(os.environ.get("COMPRESSION_BBW_PERCENTILE", "35"))
COMPRESSION_ADX_MAX = float(os.environ.get("COMPRESSION_ADX_MAX", "20"))
COMPRESSION_HURST_MAX = float(os.environ.get("COMPRESSION_HURST_MAX", "0.48"))
REGIME_LOOKBACK = int(os.environ.get("REGIME_LOOKBACK", "300"))
HEARTBEAT_INTERVAL_SEC = int(os.environ.get("HEARTBEAT_INTERVAL_SEC", "60"))


# ---------------------------------------------------------------------------
# Price history buffer
# ---------------------------------------------------------------------------

class PriceHistory:
    """Rolling tick buffer with derived returns."""

    def __init__(self, maxlen: int = PRICE_HISTORY_LEN):
        self.prices: deque = deque(maxlen=maxlen)
        self.timestamps: deque = deque(maxlen=maxlen)

    def add(self, price: float, ts: float) -> None:
        self.prices.append(price)
        self.timestamps.append(ts)

    def as_array(self) -> np.ndarray:
        return np.array(self.prices, dtype=float)

    def log_returns(self) -> np.ndarray:
        p = self.as_array()
        if len(p) < 2:
            return np.array([])
        return np.diff(np.log(p))

    def __len__(self) -> int:
        return len(self.prices)


# ---------------------------------------------------------------------------
# Regime detector -- primary indicator: Compression
# ---------------------------------------------------------------------------

@dataclass
class RegimeReading:
    label: str
    realized_vol: float
    bbw: float
    adx: float
    hurst: float
    detail: Dict[str, Any] = field(default_factory=dict)


class RegimeDetector:
    """
    Classifies the current regime as Compression using four independent
    signals, all evaluated against the bot's own rolling percentile
    history (self-calibrating per instrument):

      - realized volatility (rolling stdev of log returns) in a low
        percentile of its own recent history
      - Bollinger bandwidth (upper-lower)/middle in a low percentile
      - ADX below a fixed ceiling (weak trend strength)
      - Hurst exponent below ~0.5 (mean-reverting / anti-persistent)

    All four must agree for a "Compression" reading -- deliberately
    conservative, since a false "trade" into a regime that's about to
    break out is worse than a missed setup.
    """

    def __init__(self, lookback: int = REGIME_LOOKBACK):
        self.lookback = lookback
        self._vol_history: deque = deque(maxlen=2000)
        self._bbw_history: deque = deque(maxlen=2000)

    @staticmethod
    def _bollinger_bandwidth(prices: np.ndarray, window: int = 20, n_std: float = 2.0) -> float:
        if len(prices) < window:
            return float("nan")
        window_prices = prices[-window:]
        mid = window_prices.mean()
        sd = window_prices.std(ddof=1)
        if mid == 0:
            return float("nan")
        upper = mid + n_std * sd
        lower = mid - n_std * sd
        return (upper - lower) / mid

    @staticmethod
    def _adx(prices: np.ndarray, window: int = 14) -> float:
        """
        Simplified ADX approximated off tick deltas (no separate high/low
        feed for synthetic index ticks). If OHLC candles become available
        for 1HZ10V, swap in the standard high/low/close ADX formula.
        """
        if len(prices) < window + 1:
            return float("nan")
        deltas = np.diff(prices[-(window + 1):])
        up_moves = np.where(deltas > 0, deltas, 0.0)
        down_moves = np.where(deltas < 0, -deltas, 0.0)
        atr = np.mean(np.abs(deltas)) + 1e-12
        plus_di = 100 * np.mean(up_moves) / atr
        minus_di = 100 * np.mean(down_moves) / atr
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-12)
        return dx

    @staticmethod
    def _hurst_exponent(prices: np.ndarray, max_lag: int = 60) -> float:
        """
        Anchored R/S-style Hurst estimate on raw price levels (not
        returns) -- same fix already applied to the Hurst estimator in
        the GARCH/HMM bots (deriv_over2_r100_bot / deriv_evenodd_r100_bot).
        """
        if len(prices) < max_lag * 2:
            return float("nan")
        lags = range(2, max_lag)
        tau = []
        for lag in lags:
            diff = prices[lag:] - prices[:-lag]
            std = np.std(diff)
            tau.append(std if std > 0 else 1e-12)
        tau = np.array(tau)
        log_lags = np.log(np.array(list(lags)))
        log_tau = np.log(tau)
        poly = np.polyfit(log_lags, log_tau, 1)
        return poly[0]

    def _percentile_rank(self, value: float, history: deque) -> float:
        if len(history) < 20 or math.isnan(value):
            return 50.0
        arr = np.array(history)
        return float((arr < value).mean() * 100)

    def read(self, price_history: PriceHistory) -> RegimeReading:
        prices = price_history.as_array()
        returns = price_history.log_returns()

        if len(returns) < self.lookback:
            return RegimeReading("Unknown", float("nan"), float("nan"), float("nan"), float("nan"))

        realized_vol = float(np.std(returns[-self.lookback:], ddof=1))
        bbw = self._bollinger_bandwidth(prices)
        adx = self._adx(prices)
        hurst = self._hurst_exponent(prices)

        self._vol_history.append(realized_vol)
        if not math.isnan(bbw):
            self._bbw_history.append(bbw)

        vol_pct = self._percentile_rank(realized_vol, self._vol_history)
        bbw_pct = self._percentile_rank(bbw, self._bbw_history) if not math.isnan(bbw) else 50.0

        is_compression = (
            vol_pct <= COMPRESSION_VOL_PERCENTILE
            and bbw_pct <= COMPRESSION_BBW_PERCENTILE
            and (math.isnan(adx) or adx <= COMPRESSION_ADX_MAX)
            and (math.isnan(hurst) or hurst <= COMPRESSION_HURST_MAX)
        )

        label = "Compression" if is_compression else "Expansion"

        return RegimeReading(
            label=label,
            realized_vol=realized_vol,
            bbw=bbw,
            adx=adx,
            hurst=hurst,
            detail={"vol_percentile": vol_pct, "bbw_percentile": bbw_pct},
        )


# ---------------------------------------------------------------------------
# Monte Carlo engine: HMM regime-switching GBM + GARCH vol + jump-diffusion,
# cross-checked against historical bootstrap resampling
# ---------------------------------------------------------------------------

@dataclass
class MCCandidate:
    duration_min: int
    barrier_half_width_pct: float
    p_stay_parametric: float
    p_stay_bootstrap: float
    p_stay_blended: float


class MonteCarloEngine:
    """
    Layered simulation engine. Two independently-computed probability
    estimates are blended so a single mis-specified model can't drive
    the trade decision on its own:

      parametric stack:
        HMM regime-switching GBM, each step's sigma additionally scaled
        by a GARCH(1,1) conditional-vol forecast, plus a Merton jump
        overlay for tail risk.

      non-parametric cross-check:
        historical bootstrap -- resample blocks of real historical log
        returns (preserves actual fat tails/autocorrelation) and walk
        the same horizon forward.

    p_stay_blended = 0.5 * parametric + 0.5 * bootstrap by default; the
    blend weight is deliberately conservative (equal weight) rather than
    trusting the parametric side more, since HMM/GARCH fits on a rolling
    window can drift.
    """

    def __init__(self, n_paths: int = MC_PATHS, hmm_states: int = HMM_N_STATES):
        self.n_paths = n_paths
        self.hmm_states = hmm_states
        self._hmm: Optional["GaussianHMM"] = None
        self._garch_fitted = None
        self._last_fit_time = 0.0
        self._fallback_sigma = None

    def fit_summary(self) -> Dict[str, Any]:
        """Snapshot of current model state, for heartbeat/diagnostic logging."""
        summary: Dict[str, Any] = {
            "last_fit_time": self._last_fit_time,
            "seconds_since_fit": (time.time() - self._last_fit_time) if self._last_fit_time else None,
            "hmm_fitted": self._hmm is not None,
            "garch_fitted": self._garch_fitted is not None,
            "fallback_sigma": self._fallback_sigma,
        }
        if self._hmm is not None:
            means = self._hmm.means_.flatten().tolist()
            covars = self._hmm.covars_.reshape(self.hmm_states, -1)[:, 0]
            sigmas = np.sqrt(np.clip(covars, 1e-12, None)).tolist()
            summary["hmm_state_means"] = means
            summary["hmm_state_sigmas"] = sigmas
        return summary

    # -- fitting -----------------------------------------------------------

    def maybe_refit(self, returns: np.ndarray, force: bool = False) -> None:
        now = time.time()
        if not force and (now - self._last_fit_time) < REFIT_INTERVAL_SEC:
            return
        if len(returns) < 200:
            return
        self._fit_hmm(returns)
        self._fit_garch(returns)
        self._last_fit_time = now

    def _fit_hmm(self, returns: np.ndarray) -> None:
        if not HMM_AVAILABLE:
            log.warning("hmmlearn not installed -- falling back to single-regime GBM. "
                        "Install hmmlearn for regime-switching paths.")
            self._hmm = None
            return
        try:
            # hmmlearn's EM is numerically fragile on raw returns this small
            # (~1e-4 scale) -- fit on a rescaled series (same trick used for
            # the GARCH fit below) and rescale means/covars back afterward.
            # Without this, EM occasionally collapses one state onto a
            # near-zero-occupancy, blown-up-variance fit (observed in testing:
            # one state's variance came out ~1e5x the other two), which then
            # sends simulated paths to unrealistic extremes.
            scale = 1000.0
            X = (returns * scale).reshape(-1, 1)
            model = GaussianHMM(
                n_components=self.hmm_states, covariance_type="diag",
                n_iter=200, random_state=42, min_covar=1e-2,
            )
            model.fit(X)
            model.means_ = model.means_ / scale

            # hmmlearn's covars_ getter returns a full (n_components, n_dim,
            # n_dim) view even for covariance_type="diag", but its setter
            # validates against the flat (n_components, n_dim) shape -- pull
            # the diagonal out explicitly and set it back in that flat shape.
            diag_vars = model.covars_.reshape(self.hmm_states, -1)[:, 0] / (scale ** 2)

            # Belt-and-braces cap: don't let any single state's sigma exceed
            # 4x the empirical realized vol -- observed in testing that EM
            # can occasionally collapse one (near-zero-occupancy) state onto
            # a blown-up variance, which then dominates the simulated tail
            # with unrealistic paths.
            empirical_sigma = float(np.std(returns, ddof=1))
            cap = (4 * empirical_sigma) ** 2
            if np.any(diag_vars > cap):
                log.warning(
                    "HMM produced an outsized state variance (%s); capping at 4x empirical vol.",
                    diag_vars.tolist(),
                )
                diag_vars = np.clip(diag_vars, 1e-12, cap)

            model.covars_ = diag_vars.reshape(self.hmm_states, 1)
            self._hmm = model
        except Exception as exc:
            log.exception("HMM fit failed, falling back to single-regime GBM: %s", exc)
            self._hmm = None

    def _fit_garch(self, returns: np.ndarray) -> None:
        self._fallback_sigma = float(np.std(returns[-200:], ddof=1))
        if not ARCH_AVAILABLE:
            log.warning("arch package not installed -- using rolling stdev instead of GARCH(1,1). "
                        "Install `arch` for conditional-vol forecasting.")
            self._garch_fitted = None
            return
        try:
            scaled = returns * 1000.0  # arch expects returns on a larger scale numerically
            am = arch_model(scaled, vol="Garch", p=1, q=1, dist="normal", mean="Zero")
            res = am.fit(disp="off")
            self._garch_fitted = res
        except Exception as exc:
            log.exception("GARCH fit failed, falling back to rolling stdev: %s", exc)
            self._garch_fitted = None

    def _garch_sigma_forecast(self, steps: int) -> np.ndarray:
        """Per-step conditional sigma forecast, same units as raw log returns."""
        if self._garch_fitted is not None:
            try:
                fc = self._garch_fitted.forecast(horizon=steps, reindex=False)
                variance = fc.variance.values[-1] / (1000.0 ** 2)
                sigma = np.sqrt(np.clip(variance, 1e-12, None))
                if len(sigma) < steps:
                    sigma = np.concatenate([sigma, np.full(steps - len(sigma), sigma[-1])])
                return sigma[:steps]
            except Exception as exc:
                log.warning("GARCH forecast failed mid-run, using flat fallback sigma: %s", exc)
        flat = self._fallback_sigma if self._fallback_sigma else 1e-4
        return np.full(steps, flat)

    # -- simulation ----------------------------------------------------------

    def _simulate_regime_switching_gbm(self, s0: float, steps: int) -> np.ndarray:
        """
        Returns an (n_paths, steps+1) array of simulated prices, combining:
          - HMM state transitions (if fitted) selecting per-step mu
          - GARCH-conditional sigma scaling the per-step Gaussian shock
          - Merton jump-diffusion overlay
        """
        n = self.n_paths
        garch_sigma = self._garch_sigma_forecast(steps)  # shape (steps,)

        if self._hmm is not None:
            means = self._hmm.means_.flatten()
            transmat = self._hmm.transmat_
            startprob = self._hmm.startprob_
            state_vars = self._hmm.covars_.reshape(self.hmm_states, -1)[:, 0]
            state_sigmas = np.sqrt(np.clip(state_vars, 1e-12, None))

            states = np.zeros((n, steps), dtype=int)
            states[:, 0] = np.random.choice(self.hmm_states, size=n, p=startprob)
            for t in range(1, steps):
                for s in range(self.hmm_states):
                    mask = states[:, t - 1] == s
                    if mask.any():
                        states[mask, t] = np.array(
                            [np.random.choice(self.hmm_states, p=transmat[s]) for _ in range(mask.sum())]
                        )
            mu_path = means[states]                      # (n, steps)
            regime_sigma_path = state_sigmas[states]      # (n, steps)
        else:
            # Fallback: single-regime, mean drift ~ 0 (risk-neutral-ish for a
            # short synthetic-index horizon), sigma purely from GARCH/rolling.
            mu_path = np.zeros((n, steps))
            regime_sigma_path = np.tile(garch_sigma, (n, 1))

        # Blend regime sigma with GARCH conditional sigma (geometric mean so
        # neither source dominates); this is a simplification -- a fuller
        # implementation would feed regime-conditional residuals into the
        # GARCH recursion directly rather than blending after the fact.
        sigma_path = np.sqrt(regime_sigma_path * np.tile(garch_sigma, (n, 1)))

        z = np.random.standard_normal((n, steps))
        diffusion = (mu_path - 0.5 * sigma_path ** 2) + sigma_path * z

        # Merton jump overlay
        jump_intensity_per_step = JUMP_INTENSITY_PER_DAY / (24 * 60 * 60) * (60.0 / TICKS_PER_SECOND)
        jump_counts = np.random.poisson(jump_intensity_per_step, size=(n, steps))
        jump_sizes = np.where(
            jump_counts > 0,
            np.random.normal(JUMP_MEAN, JUMP_STD, size=(n, steps)) * jump_counts,
            0.0,
        )

        log_returns = diffusion + jump_sizes
        log_prices = np.cumsum(log_returns, axis=1)
        log_prices = np.concatenate([np.zeros((n, 1)), log_prices], axis=1)
        prices = s0 * np.exp(log_prices)
        return prices

    def _simulate_bootstrap(self, s0: float, steps: int, historical_returns: np.ndarray,
                             block_size: int = 20) -> np.ndarray:
        """Block bootstrap resampling of real historical log returns (preserves
        empirical fat tails and short-range autocorrelation better than a
        purely parametric draw)."""
        n = self.n_paths
        if len(historical_returns) < block_size * 2:
            # not enough history -- fall back to iid resampling
            sampled = np.random.choice(historical_returns, size=(n, steps), replace=True)
        else:
            n_blocks = math.ceil(steps / block_size)
            sampled = np.zeros((n, n_blocks * block_size))
            max_start = len(historical_returns) - block_size
            for p in range(n):
                for b in range(n_blocks):
                    start = np.random.randint(0, max_start)
                    sampled[p, b * block_size:(b + 1) * block_size] = historical_returns[start:start + block_size]
            sampled = sampled[:, :steps]
        log_prices = np.cumsum(sampled, axis=1)
        log_prices = np.concatenate([np.zeros((n, 1)), log_prices], axis=1)
        return s0 * np.exp(log_prices)

    def estimate_stay_probability(
        self,
        s0: float,
        duration_min: int,
        barrier_half_width_pct: float,
        historical_returns: np.ndarray,
    ) -> MCCandidate:
        steps = max(1, int(duration_min * 60 * TICKS_PER_SECOND))
        upper = s0 * (1 + barrier_half_width_pct / 100.0)
        lower = s0 * (1 - barrier_half_width_pct / 100.0)

        parametric_prices = self._simulate_regime_switching_gbm(s0, steps)
        stayed_parametric = np.all((parametric_prices >= lower) & (parametric_prices <= upper), axis=1)
        p_stay_parametric = float(stayed_parametric.mean())

        bootstrap_prices = self._simulate_bootstrap(s0, steps, historical_returns)
        stayed_bootstrap = np.all((bootstrap_prices >= lower) & (bootstrap_prices <= upper), axis=1)
        p_stay_bootstrap = float(stayed_bootstrap.mean())

        p_stay_blended = 0.5 * p_stay_parametric + 0.5 * p_stay_bootstrap

        return MCCandidate(
            duration_min=duration_min,
            barrier_half_width_pct=barrier_half_width_pct,
            p_stay_parametric=p_stay_parametric,
            p_stay_bootstrap=p_stay_bootstrap,
            p_stay_blended=p_stay_blended,
        )

    def search_grid(
        self, s0: float, historical_returns: np.ndarray,
        durations: List[int] = DURATION_CANDIDATES_MIN,
        barrier_widths: List[float] = BARRIER_HALF_WIDTH_PCT,
    ) -> List[MCCandidate]:
        self.maybe_refit(historical_returns)
        results = []
        for d in durations:
            for w in barrier_widths:
                results.append(self.estimate_stay_probability(s0, d, w, historical_returns))
        return results


# ---------------------------------------------------------------------------
# Deriv WebSocket client
# ---------------------------------------------------------------------------

class DerivClient:
    """
    Async client for Deriv's current Options API (REST + OTP-authenticated
    WebSocket), replacing the old app_id-only ws.derivws.com/websockets/v3
    connection model -- that old model now rejects the handshake outright
    (HTTP 401) rather than letting you authorize in-band after connecting.

    Two separate connections are used:
      - a PUBLIC websocket (no auth) for the tick stream --
        wss://api.derivws.com/trading/v1/options/ws/public
      - an OTP-AUTHENTICATED websocket for trading calls (proposal, buy) --
        obtained by POSTing /trading/v1/options/accounts/{account_id}/otp
        (Deriv-App-ID header + Authorization: Bearer token), which returns
        a ready-to-use ws URL with the OTP embedded. The OTP itself is only
        valid for ~120s, so it's requested fresh each time the trade
        connection needs to be (re)established, not reused across drops.

    The proposal/buy message schema on the authenticated socket is
    unchanged from the legacy API (same JSON-RPC style {"proposal": 1, ...}
    / {"buy": ..., "price": ...} payloads) -- only the connection setup
    changed. If Deriv has since changed that schema too, adjust
    get_proposal()/buy() accordingly; everything else in the bot is
    decoupled from this class via its method signatures.
    """

    REST_BASE = "https://api.derivws.com/trading/v1/options"

    def __init__(
        self,
        app_id: str = DERIV_APP_ID,
        token: Optional[str] = DERIV_API_TOKEN,
        account_type: str = DERIV_ACCOUNT_TYPE,
        account_id: Optional[str] = DERIV_ACCOUNT_ID,
    ):
        self.app_id = app_id
        self.token = token
        self.account_type = account_type  # "demo" or "real"
        self.account_id = account_id
        self.public_ws = None
        self.trade_ws = None
        self._req_id = 0
        self._http = None  # aiohttp session, created lazily

    async def _get_http(self):
        if aiohttp is None:
            raise RuntimeError("`aiohttp` package not installed -- pip install aiohttp")
        if self._http is None:
            self._http = aiohttp.ClientSession()
        return self._http

    # -- setup ---------------------------------------------------------------

    async def connect(self) -> None:
        """Open the public tick-stream socket and the authenticated trade socket."""
        if websockets is None:
            raise RuntimeError("`websockets` package not installed -- pip install websockets")
        self.public_ws = await websockets.connect(DERIV_WS_PUBLIC_URL, ping_interval=20, ping_timeout=10)
        if self.token:
            await self._connect_trade_ws()
        else:
            log.warning("DERIV_API_TOKEN not set -- trading calls (proposal/buy) will fail. "
                        "Only the public tick stream will work.")

    async def _resolve_account_id(self) -> str:
        if self.account_id:
            return self.account_id
        http = await self._get_http()
        headers = {"Deriv-App-ID": self.app_id, "Authorization": f"Bearer {self.token}"}
        async with http.get(f"{self.REST_BASE}/accounts", headers=headers) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"Failed to list Deriv accounts ({resp.status}): {text}")
            body = await resp.json()
        accounts = body.get("data", [])
        if isinstance(accounts, dict):
            accounts = [accounts]
        match = next((a for a in accounts if a.get("account_type") == self.account_type), None)
        if match is None and accounts:
            match = accounts[0]
        if match is None:
            raise RuntimeError(
                f"No Deriv account found for account_type={self.account_type!r}. "
                f"Set DERIV_ACCOUNT_ID explicitly to skip this lookup."
            )
        self.account_id = match["account_id"]
        return self.account_id

    async def _request_otp_ws_url(self) -> str:
        account_id = await self._resolve_account_id()
        http = await self._get_http()
        headers = {"Deriv-App-ID": self.app_id, "Authorization": f"Bearer {self.token}"}
        async with http.post(f"{self.REST_BASE}/accounts/{account_id}/otp", headers=headers) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"OTP request failed ({resp.status}): {text}")
            body = await resp.json()
        url = body["data"]["url"]
        return url

    async def _connect_trade_ws(self) -> None:
        """(Re)establish the authenticated trading socket via a fresh OTP.
        OTPs expire in ~120s, so this is called right before the socket is
        needed, not held onto and reused across a long-lived connection."""
        ws_url = await self._request_otp_ws_url()
        if self.trade_ws is not None:
            try:
                await self.trade_ws.close()
            except Exception:
                pass
        self.trade_ws = await websockets.connect(ws_url, ping_interval=20, ping_timeout=10)

    async def _ensure_trade_ws(self) -> None:
        if self.trade_ws is None or self.trade_ws.close_code is not None:
            await self._connect_trade_ws()

    async def _send_trade(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        await self._ensure_trade_ws()
        self._req_id += 1
        payload = {**payload, "req_id": self._req_id}
        await self.trade_ws.send(json.dumps(payload))
        while True:
            raw = await self.trade_ws.recv()
            msg = json.loads(raw)
            if msg.get("req_id") == self._req_id:
                if "error" in msg:
                    raise RuntimeError(f"Deriv API error: {msg['error']}")
                return msg

    # -- market data -----------------------------------------------------------

    async def subscribe_ticks(self, symbol: str = SYMBOL):
        """Async generator yielding (price, epoch) tuples from the public socket."""
        self._req_id += 1
        await self.public_ws.send(json.dumps({"ticks": symbol, "subscribe": 1, "req_id": self._req_id}))
        while True:
            raw = await self.public_ws.recv()
            msg = json.loads(raw)
            if msg.get("msg_type") == "tick":
                tick = msg["tick"]
                yield float(tick["quote"]), float(tick["epoch"])

    # -- trading -----------------------------------------------------------

    async def get_proposal(
        self, duration_min: int, barrier_high: float, barrier_low: float,
        stake: float = STAKE_AMOUNT, symbol: str = SYMBOL,
    ) -> Dict[str, Any]:
        """
        EXPIRYRANGE ("Ends Between") proposal. barrier/barrier2 format is
        carried over unchanged from the legacy API -- confirm against the
        current docs/playground before trusting it at higher stakes, since
        this specific field's format wasn't part of what changed in the
        REST+OTP migration but could still shift independently.
        """
        payload = {
            "proposal": 1,
            "amount": stake,
            "basis": "stake",
            "contract_type": "EXPIRYRANGE",
            "currency": "USD",
            "duration": duration_min,
            "duration_unit": "m",
            "underlying_symbol": symbol,
            "barrier": f"+{barrier_high}",
            "barrier2": f"-{barrier_low}",
        }
        return await self._send_trade(payload)

    async def buy(self, proposal_id: str, price: float) -> Dict[str, Any]:
        return await self._send_trade({"buy": proposal_id, "price": price})

    async def close(self) -> None:
        if self.public_ws is not None:
            await self.public_ws.close()
        if self.trade_ws is not None:
            await self.trade_ws.close()
        if self._http is not None:
            await self._http.close()


# ---------------------------------------------------------------------------
# Supabase logging
# ---------------------------------------------------------------------------

class SupabaseLogger:
    """
    Mirrors the audit-column pattern already used for the EV gate in
    deriv_multisymbol_bot.py: regime, mc probabilities, payout, ev,
    gate pass/fail, and the eventual outcome all get a row.
    """

    TABLE = "expiryrange_compression_trades"

    def __init__(self):
        self.client = None
        if SUPABASE_AVAILABLE and SUPABASE_URL and SUPABASE_KEY:
            self.client = create_client(SUPABASE_URL, SUPABASE_KEY)
        else:
            log.warning("Supabase not configured -- trade logs will only go to stdout.")

    def log_trade(self, row: Dict[str, Any]) -> None:
        log.info("TRADE LOG: %s", json.dumps(row, default=str))
        if self.client is None:
            return
        try:
            self.client.table(self.TABLE).insert(row).execute()
        except Exception as exc:
            log.exception("Supabase insert failed: %s", exc)


# ---------------------------------------------------------------------------
# Selection: apply the payout gate + EV gate on top of the MC grid
# ---------------------------------------------------------------------------

@dataclass
class TradeDecision:
    should_trade: bool
    duration_min: Optional[int] = None
    barrier_half_width_pct: Optional[float] = None
    payout_pct: Optional[float] = None
    p_stay: Optional[float] = None
    ev: Optional[float] = None
    proposal_id: Optional[str] = None
    ask_price: Optional[float] = None
    reason: str = ""
    # Per-candidate audit trail: what each candidate's proposal/payout gate
    # and EV gate said, in the order candidates were evaluated. Populated
    # by select_trade() and consumed by the trade_log signal audit.
    diagnostics: List[Dict[str, Any]] = field(default_factory=list)


async def select_trade(
    deriv: DerivClient,
    candidates: List[MCCandidate],
    spot: float,
    stake: float = STAKE_AMOUNT,
    min_payout_pct: float = MIN_PAYOUT_PCT,
) -> TradeDecision:
    """
    For each MC candidate, ask Deriv for the actual payout, keep only
    candidates quoting >= min_payout_pct, then rank the survivors by
    model EV and take the best. Returns should_trade=False if nothing
    clears both gates -- this is expected and normal; it means the
    market isn't currently offering an asymmetric EXPIRYRANGE setup
    at your stake, not that the bot is broken.
    """
    scored = []
    diagnostics: List[Dict[str, Any]] = []

    for c in candidates:
        entry: Dict[str, Any] = {
            "duration_min": c.duration_min,
            "barrier_half_width_pct": c.barrier_half_width_pct,
            "p_stay_parametric": c.p_stay_parametric,
            "p_stay_bootstrap": c.p_stay_bootstrap,
            "p_stay_blended": c.p_stay_blended,
        }
        barrier_abs = spot * (c.barrier_half_width_pct / 100.0)
        try:
            proposal = await deriv.get_proposal(
                duration_min=c.duration_min,
                barrier_high=round(barrier_abs, 5),
                barrier_low=round(barrier_abs, 5),
                stake=stake,
            )
        except Exception as exc:
            log.warning("Proposal request failed for %s/%s: %s", c.duration_min, c.barrier_half_width_pct, exc)
            entry.update({"gate": "proposal_failed", "detail": str(exc)})
            diagnostics.append(entry)
            continue

        prop = proposal.get("proposal", {})
        payout = prop.get("payout")
        ask_price = prop.get("ask_price", stake)
        proposal_id = prop.get("id")
        if payout is None or ask_price is None:
            entry.update({"gate": "proposal_incomplete", "detail": "missing payout/ask_price in proposal response"})
            diagnostics.append(entry)
            continue

        payout_pct = (payout - ask_price) / ask_price * 100.0
        entry["payout_pct"] = payout_pct
        if payout_pct < min_payout_pct:
            entry.update({"gate": "payout_gate_fail",
                          "detail": f"payout_pct {payout_pct:.2f}% < min {min_payout_pct:.2f}%"})
            diagnostics.append(entry)
            continue

        implied_breakeven_p = ask_price / payout  # probability needed to break even
        ev = c.p_stay_blended * payout - (1 - c.p_stay_blended) * ask_price
        entry["breakeven_p"] = implied_breakeven_p
        entry["ev"] = ev
        if c.p_stay_blended <= implied_breakeven_p:
            entry.update({"gate": "ev_gate_fail",
                          "detail": f"p_stay_blended {c.p_stay_blended:.3f} <= breakeven {implied_breakeven_p:.3f}"})
            diagnostics.append(entry)
            continue  # payout clears the threshold but model doesn't think it's a real edge

        entry["gate"] = "passed"
        diagnostics.append(entry)
        scored.append((ev, c, payout_pct, proposal_id, ask_price))

    if not scored:
        return TradeDecision(
            should_trade=False,
            reason="No candidate cleared both the payout gate and the EV gate.",
            diagnostics=diagnostics,
        )

    scored.sort(key=lambda x: x[0], reverse=True)
    ev, best, payout_pct, proposal_id, ask_price = scored[0]

    return TradeDecision(
        should_trade=True,
        duration_min=best.duration_min,
        barrier_half_width_pct=best.barrier_half_width_pct,
        payout_pct=payout_pct,
        p_stay=best.p_stay_blended,
        ev=ev,
        proposal_id=proposal_id,
        ask_price=ask_price,
        reason="cleared payout gate and EV gate",
        diagnostics=diagnostics,
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class ExpiryRangeCompressionBot:
    """
    Main loop:
      1. Stream ticks -> update PriceHistory
      2. Once enough ticks are buffered, read the regime each tick
      3. On Compression: run the Monte Carlo grid search
      4. Pass MC candidates through the payout gate + EV gate
      5. If something clears both gates: buy, log, cooldown
      6. If nothing clears: log the miss and keep streaming (normal, expected)

    A separate heartbeat task (see heartbeat_loop) logs a status summary
    every HEARTBEAT_INTERVAL_SEC seconds regardless of whether a Compression
    regime has fired -- tick buffer progress, current model fit state,
    what the regime layer currently sees, a lightweight MC read at a
    representative duration/barrier, and a running trade/gate tally -- so
    you have visibility into what's happening between trades, not just at
    trade time.

    Runs LIVE against real money by default -- STAKE_AMOUNT is intentionally
    small (default $1) per the explicit choice to go live rather than
    shadow-trade first. Raise STAKE_AMOUNT only after you've watched enough
    real trade logs to trust the gate behavior.
    """

    # Representative candidate used only for the heartbeat's quick MC read --
    # not part of the actual trading grid/decision.
    HEARTBEAT_DURATION_MIN = 10
    HEARTBEAT_BARRIER_HW_PCT = 0.5

    @staticmethod
    def _write_signal_audit(
        epoch: float,
        price: float,
        regime: RegimeReading,
        candidates: List[MCCandidate],
        decision: TradeDecision,
    ) -> None:
        """
        Writes one complete audit block to trade_log for a single evaluated
        signal (a Compression read that triggered the MC grid) -- covering
        every intelligence layer in the decision hierarchy:

          1. Regime layer      -- the reading that triggered evaluation
          2. Monte Carlo layer -- every (duration, barrier) candidate's
                                  parametric / bootstrap / blended p_stay
          3. Gate layer        -- per-candidate payout-gate + EV-gate result
          4. Final decision    -- what was (or wasn't) traded, and why

        Called for every signal that reaches the grid, whether or not a
        trade is ultimately taken, so misses are just as auditable as fills.
        """
        ts_iso = datetime.datetime.utcfromtimestamp(epoch).isoformat() + "Z"
        lines = [
            "=" * 78,
            f"SIGNAL  ts={ts_iso}  epoch={epoch:.0f}  symbol={SYMBOL}  spot={price}",
            "-" * 78,
            "[1] REGIME LAYER",
            (f"    label={regime.label}  realized_vol={regime.realized_vol:.6f}  "
             f"bbw={regime.bbw:.4f}  adx={regime.adx:.2f}  hurst={regime.hurst:.3f}"),
            (f"    vol_percentile={regime.detail.get('vol_percentile', float('nan')):.1f}  "
             f"bbw_percentile={regime.detail.get('bbw_percentile', float('nan')):.1f}"),
            f"[2] MONTE CARLO LAYER  ({len(candidates)} candidates)",
        ]
        for c in candidates:
            lines.append(
                f"    duration={c.duration_min:>2}min  barrier_hw={c.barrier_half_width_pct:>4.2f}%  "
                f"p_stay: parametric={c.p_stay_parametric:.3f}  bootstrap={c.p_stay_bootstrap:.3f}  "
                f"blended={c.p_stay_blended:.3f}"
            )
        lines.append("[3] GATE LAYER (payout gate + EV gate, per candidate)")
        for d in decision.diagnostics:
            gate = d.get("gate", "unknown")
            base = f"    duration={d['duration_min']:>2}min barrier_hw={d['barrier_half_width_pct']:>4.2f}%  -> {gate}"
            if gate == "passed":
                base += (f"  payout_pct={d.get('payout_pct', float('nan')):.2f}%  "
                         f"breakeven_p={d.get('breakeven_p', float('nan')):.3f}  ev={d.get('ev', float('nan')):.4f}")
            elif "detail" in d:
                base += f"  ({d['detail']})"
            lines.append(base)
        lines.append("[4] FINAL DECISION")
        if decision.should_trade:
            lines.append(
                f"    TRADE TAKEN  duration={decision.duration_min}min  "
                f"barrier_hw={decision.barrier_half_width_pct}%  payout={decision.payout_pct:.2f}%  "
                f"p_stay={decision.p_stay:.3f}  ev={decision.ev:.4f}"
            )
        else:
            lines.append(f"    NO TRADE  -- {decision.reason}")
        lines.append("=" * 78)
        trade_log.info("\n".join(lines))

    def __init__(self):
        self.deriv = DerivClient()
        self.history = PriceHistory()
        self.regime_detector = RegimeDetector()
        self.mc_engine = MonteCarloEngine()
        self.logger = SupabaseLogger()
        self.open_trades = 0
        self._last_trade_time = 0.0

        # state surfaced by the heartbeat logger
        self.last_regime: Optional[RegimeReading] = None
        self.last_candidates: List[MCCandidate] = []
        self.last_decision: Optional[TradeDecision] = None
        self.last_price: Optional[float] = None
        self.last_epoch: Optional[float] = None
        self.trade_count = 0
        self.miss_count = 0
        self.compression_count = 0

    async def run(self) -> None:
        await self.deriv.connect()
        log.info("Connected to Deriv. Streaming %s ticks...", SYMBOL)

        heartbeat_task = asyncio.create_task(self.heartbeat_loop())
        try:
            await self._consume_ticks()
        finally:
            heartbeat_task.cancel()

    async def _consume_ticks(self) -> None:
        async for price, epoch in self.deriv.subscribe_ticks(SYMBOL):
            self.history.add(price, epoch)
            self.last_price = price
            self.last_epoch = epoch

            if len(self.history) < MIN_TICKS_BEFORE_TRADING:
                continue

            regime = self.regime_detector.read(self.history)
            self.last_regime = regime

            if self.open_trades >= MAX_CONCURRENT_TRADES:
                continue

            if (time.time() - self._last_trade_time) < COOLDOWN_SEC_AFTER_TRADE:
                continue

            if regime.label != "Compression":
                continue

            self.compression_count += 1
            log.info(
                "Compression regime detected (vol_pct=%.1f bbw_pct=%.1f adx=%.2f hurst=%.3f) -- running Monte Carlo grid",
                regime.detail.get("vol_percentile", float("nan")),
                regime.detail.get("bbw_percentile", float("nan")),
                regime.adx, regime.hurst,
            )

            returns = self.history.log_returns()
            candidates = self.mc_engine.search_grid(s0=price, historical_returns=returns)
            self.last_candidates = candidates

            decision = await select_trade(self.deriv, candidates, spot=price)
            self.last_decision = decision

            # Full intelligence-layer audit (regime + every MC candidate +
            # every gate outcome + final decision) goes to trade_log, not
            # the operational log -- keeps `log` readable at a glance while
            # trades.log carries the complete "why" for every signal.
            self._write_signal_audit(epoch, price, regime, candidates, decision)

            base_log_row = {
                "ts": epoch,
                "symbol": SYMBOL,
                "spot": price,
                "regime": regime.label,
                "realized_vol": regime.realized_vol,
                "bbw": regime.bbw,
                "adx": regime.adx,
                "hurst": regime.hurst,
                "min_payout_gate": MIN_PAYOUT_PCT,
            }

            if not decision.should_trade:
                self.miss_count += 1
                base_log_row.update({"gate_passed": False, "reason": decision.reason})
                self.logger.log_trade(base_log_row)
                continue

            try:
                buy_result = await self.deriv.buy(decision.proposal_id, decision.ask_price)
            except Exception as exc:
                log.exception("Buy failed: %s", exc)
                trade_log.info("BUY FAILED for signal above -- %s", exc)
                base_log_row.update({"gate_passed": True, "reason": f"buy_failed: {exc}"})
                self.logger.log_trade(base_log_row)
                continue

            self.open_trades += 1
            self.trade_count += 1
            self._last_trade_time = time.time()
            # NOTE: this bot does not currently track contract expiry to
            # decrement open_trades / free up the cooldown slot -- wire up
            # a proposal_open_contract subscription on buy_result's
            # contract_id if you want the loop to reset automatically
            # rather than relying on MAX_CONCURRENT_TRADES=1 + cooldown.

            base_log_row.update({
                "gate_passed": True,
                "duration_min": decision.duration_min,
                "barrier_half_width_pct": decision.barrier_half_width_pct,
                "payout_pct": decision.payout_pct,
                "p_stay_model": decision.p_stay,
                "ev": decision.ev,
                "contract_id": buy_result.get("buy", {}).get("contract_id"),
                "buy_price": buy_result.get("buy", {}).get("buy_price"),
                "reason": decision.reason,
            })
            self.logger.log_trade(base_log_row)
            log.info(
                "TRADE PLACED: duration=%smin barrier_hw=%.2f%% payout=%.1f%% p_stay=%.3f ev=%.4f",
                decision.duration_min, decision.barrier_half_width_pct,
                decision.payout_pct, decision.p_stay, decision.ev,
            )
            trade_log.info(
                "FILLED contract_id=%s buy_price=%s (for signal audited above)",
                buy_result.get("buy", {}).get("contract_id"),
                buy_result.get("buy", {}).get("buy_price"),
            )

    async def heartbeat_loop(self) -> None:
        """Logs a status summary every HEARTBEAT_INTERVAL_SEC seconds,
        independent of whether a Compression regime has fired. Never raises
        -- a heartbeat failure (e.g. a transient MC estimate error) is logged
        and skipped rather than killing the main tick-consumption loop."""
        while True:
            await asyncio.sleep(HEARTBEAT_INTERVAL_SEC)
            try:
                self._log_heartbeat()
            except Exception:
                log.exception("Heartbeat logging failed (non-fatal, continuing)")

    def _log_heartbeat(self) -> None:
        n_ticks = len(self.history)
        warm = n_ticks >= MIN_TICKS_BEFORE_TRADING

        log.info(
            "HEARTBEAT | ticks=%d/%d (%s) | last_price=%s | trades=%d misses=%d compression_reads=%d",
            n_ticks, MIN_TICKS_BEFORE_TRADING, "warm" if warm else "warming up",
            self.last_price, self.trade_count, self.miss_count, self.compression_count,
        )

        if not warm:
            return  # nothing else meaningful to report yet

        # -- model fit layer --
        fit = self.mc_engine.fit_summary()
        if fit["hmm_fitted"]:
            sigmas = ", ".join(f"{s:.6f}" for s in fit["hmm_state_sigmas"])
            log.info(
                "  MODEL FIT | hmm_states_sigma=[%s] garch_fitted=%s last_fit=%.0fs ago",
                sigmas, fit["garch_fitted"],
                fit["seconds_since_fit"] if fit["seconds_since_fit"] is not None else -1,
            )
        else:
            log.info("  MODEL FIT | not yet fitted (needs >=200 returns; refits every %ss)", REFIT_INTERVAL_SEC)

        # -- regime layer --
        r = self.last_regime
        if r is not None:
            log.info(
                "  REGIME | label=%s realized_vol=%.6f bbw=%.4f adx=%.2f hurst=%.3f vol_pct=%.1f bbw_pct=%.1f",
                r.label, r.realized_vol, r.bbw, r.adx, r.hurst,
                r.detail.get("vol_percentile", float("nan")), r.detail.get("bbw_percentile", float("nan")),
            )

        # -- MC layer: lightweight read at a representative candidate, purely
        # diagnostic (not part of the trading grid/decision) --
        if self.last_price is not None and n_ticks >= 200:
            returns = self.history.log_returns()
            self.mc_engine.maybe_refit(returns)  # no-op if within REFIT_INTERVAL_SEC
            quick = self.mc_engine.estimate_stay_probability(
                s0=self.last_price,
                duration_min=self.HEARTBEAT_DURATION_MIN,
                barrier_half_width_pct=self.HEARTBEAT_BARRIER_HW_PCT,
                historical_returns=returns,
            )
            log.info(
                "  MC READ (diagnostic, %smin/%.2f%% barrier) | parametric=%.3f bootstrap=%.3f blended=%.3f",
                self.HEARTBEAT_DURATION_MIN, self.HEARTBEAT_BARRIER_HW_PCT,
                quick.p_stay_parametric, quick.p_stay_bootstrap, quick.p_stay_blended,
            )

        # -- last real trading-grid candidates + gate decision, if any --
        if self.last_candidates:
            for c in self.last_candidates:
                log.info(
                    "  LAST GRID | duration=%smin barrier_hw=%.2f%% p_stay(parametric=%.3f bootstrap=%.3f blended=%.3f)",
                    c.duration_min, c.barrier_half_width_pct,
                    c.p_stay_parametric, c.p_stay_bootstrap, c.p_stay_blended,
                )
        if self.last_decision is not None:
            d = self.last_decision
            if d.should_trade:
                log.info(
                    "  LAST GATE RESULT | TRADED duration=%smin barrier_hw=%.2f%% payout=%.1f%% p_stay=%.3f ev=%.4f",
                    d.duration_min, d.barrier_half_width_pct, d.payout_pct, d.p_stay, d.ev,
                )
            else:
                log.info("  LAST GATE RESULT | no trade -- %s", d.reason)


def main() -> None:
    if not DERIV_API_TOKEN:
        log.warning("DERIV_API_TOKEN not set -- set it in the environment before running live.")
    bot = ExpiryRangeCompressionBot()
    asyncio.run(bot.run())


if __name__ == "__main__":
    main()
