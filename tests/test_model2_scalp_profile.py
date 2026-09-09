"""Validate the shipped profile without using any machine-local credentials."""
from __future__ import annotations

from pathlib import Path
import shutil
from types import SimpleNamespace
from dataclasses import replace

import pytest

from btc_futures_bot.costs import CostConfig
from btc_futures_bot.engine import EngineConfig, TradingEngine
from btc_futures_bot.macro_risk import MacroRiskConfig
from btc_futures_bot.main import load_config
from btc_futures_bot.risk import RiskConfig, RiskManager
from btc_futures_bot.strategy import MultiTimeframeStrategy, StrategyConfig, dynamic_stop_loss_pct
from btc_futures_bot.models import Candle
from btc_futures_bot.trade_model import EntryGate, MetaModelConfig
from btc_futures_bot.trade_model.policy import build_execution_policy


@pytest.fixture
def profile(tmp_path):
    root = Path(__file__).resolve().parents[1]
    for name in ("config.binance.model2.json", "config.binance.testnet.json"):
        shutil.copyfile(root / name, tmp_path / name)
    return load_config(tmp_path / "config.binance.model2.json")


def test_shipped_scalp_profile_has_matched_short_horizon(profile):
    strategy = StrategyConfig(**profile["strategy"])
    policy = build_execution_policy(profile)
    assert profile["mode"] == "paper"
    assert profile["active_exchange"] == "okx"
    assert profile["exchanges"]["okx"]["environment"] == "demo"
    assert strategy.mode == "scalp_v2"
    assert (strategy.trigger_timeframe, strategy.regime_timeframe) == ("1m", "5m")
    assert strategy.enable_time_exit
    assert strategy.scalp_cost_filter_enabled
    assert strategy.scalp_cost_lookback_windows == 6
    assert (strategy.max_hold_seconds, strategy.hard_max_hold_seconds) == (300, 600)
    assert policy["history_window"]["closed_history_limit"] == 99
    assert policy["labeling"]["barrier_config"]["horizon_bars"] == 10
    assert policy["labeling"]["barrier_config"]["take_profit_pct"] == pytest.approx(0.00375)
    assert profile["trade_model"]["require_approved_for_live"] is True
    assert "trade_model_2_0_scalp" in profile["trade_model"]["artifact_path"]


@pytest.mark.parametrize("side", ["long", "short"])
def test_tightest_scalp_stop_still_passes_unchanged_cost_floor(profile, side):
    strategy = StrategyConfig(**profile["strategy"])
    risk = RiskManager(RiskConfig(**profile["risk"]), costs=CostConfig(**profile["costs"]))
    candles = [Candle(i * 60_000, 100, 100.01, 99.99, 100, 1) for i in range(30)]
    stop = dynamic_stop_loss_pct(candles, strategy, risk.config.stop_loss_pct,
                                 cost_round_trip_pct=risk.costs.round_trip_pct)
    protection = risk.protection(side, 10000, 100, strategy.take_profit_r, stop)
    assert stop == pytest.approx(0.0021)
    assert risk.costs.min_net_edge_pct == 0.0015
    assert risk.costs.round_trip_pct == pytest.approx(0.0014)
    assert risk.is_cost_effective(side, 100, protection.take_profit_price, protection.quantity)
    assert risk.config.max_daily_loss_pct == 0.02
    assert risk.config.max_notional_pct == 0.3


def test_scalp_macro_floor_does_not_disable_calendar(profile):
    macro = MacroRiskConfig.from_mapping(profile["macro_event_risk"])
    assert macro.enabled and macro.events
    assert macro.shock_min_range_pct == 0.0015
    assert macro.shock_cooldown_minutes == 3
    assert macro.shock_entry_policy == "hard_block"


@pytest.mark.parametrize("side", ["long", "short"])
def test_shipped_model_shadow_allows_real_scalp_paper_entry(profile, tmp_path, side):
    pytest.importorskip("lightgbm")
    root = Path(__file__).resolve().parents[1]
    model = profile["trade_model"]
    model["artifact_path"] = str(root / model["artifact_path"])
    model["manifest_path"] = str(root / model["manifest_path"])
    model["decision_log_path"] = str(tmp_path / "decisions.sqlite3")
    assert model["mode"] == "shadow"
    gate = EntryGate(MetaModelConfig.from_mapping(profile, tmp_path))
    try:
        assert gate.status()["ready"], gate.status()["error"]
        closes = [100 + .03 * i + (.12 if i % 2 == 0 else -.12) for i in range(80)]
        final_close = closes[-1] + .65
        asof = 1_780_002_000_000
        market = {}
        for name, interval in (("1m", 60_000), ("5m", 300_000), ("1h", 3_600_000)):
            last_closed = asof // interval * interval - interval
            bars = []
            for i, close in enumerate(closes):
                opening = closes[i - 1] if i else close
                bars.append(Candle(last_closed - (80 - i) * interval, opening,
                                   max(opening, close) + .03, min(opening, close) - .03,
                                   close, 10))
            bars.append(Candle(last_closed, closes[-1], final_close + .03, closes[-1] - .03, final_close, 15))
            # Adapter includes one forming bar, which the engine must drop.
            bars.append(Candle(last_closed + interval, final_close, final_close + .03, final_close - .03, final_close, 10))
            if side == "short":
                bars = [replace(c, open=200-c.open, high=200-c.low,
                                low=200-c.high, close=200-c.close) for c in bars]
            market[name] = bars

        class PaperOnlyAdapter:
            name = "okx"
            settings = SimpleNamespace(symbol="BTC-USDT-SWAP", environment="demo")

            def fetch_candles(self, interval, limit):
                return market[interval]

            def place_market_order(self, *args, **kwargs):
                pytest.fail("paper must never submit an exchange order")

        engine = TradingEngine(
            PaperOnlyAdapter(), MultiTimeframeStrategy(StrategyConfig(**profile["strategy"]), costs=CostConfig(**profile["costs"])),
            RiskManager(RiskConfig(**profile["risk"]), costs=CostConfig(**profile["costs"])),
            EngineConfig(mode="paper", candle_limit=100, take_profit_r=1.5), entry_gate=gate,
        )
        result = engine.evaluate_once()
        assert result.status == "paper_signal", result
        assert engine.position is not None and engine.position.side == side
        assert result.signal.meta_decision.startswith("shadow_")
        assert gate.status()["recorded_candidates"] == 1
        # Repeated polling cannot open or score this minute's candidate again.
        assert engine.evaluate_once().status == "no_action"
        assert gate.status()["recorded_candidates"] == 1
    finally:
        gate.close()
