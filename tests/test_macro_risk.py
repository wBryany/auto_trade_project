from __future__ import annotations

import time
from datetime import datetime, timezone
from types import SimpleNamespace

from btc_futures_bot.engine import EngineConfig, TradingEngine
from btc_futures_bot.macro_risk import (
    MacroEvent,
    MacroRiskConfig,
    MacroRiskController,
    MacroRiskDecision,
)
from btc_futures_bot.models import Candle, Position, Signal
from btc_futures_bot.risk import RiskManager
from btc_futures_bot.strategy import StrategyConfig


def test_configured_fomc_window_blocks_entries() -> None:
    event_ms = int(datetime(2026, 7, 29, 18, 0, tzinfo=timezone.utc).timestamp() * 1000)
    config = MacroRiskConfig(
        enabled=True,
        bls_ics_url="",
        events=(MacroEvent("FOMC", event_ms, pre_minutes=240, post_minutes=120),),
    )
    controller = MacroRiskController(config)

    decision = controller.decision({}, now_ms=event_ms - 60 * 60_000)

    assert decision.blocked is True
    assert decision.reason == "macro_event:FOMC"
    assert decision.seconds_to_event == 3600


def test_bls_calendar_keeps_only_configured_high_impact_releases() -> None:
    content = """BEGIN:VCALENDAR
BEGIN:VEVENT
DTSTART;TZID=America/New_York:20260807T083000
SUMMARY:Employment Situation for July 2026
END:VEVENT
BEGIN:VEVENT
DTSTART;TZID=America/New_York:20260812T083000
SUMMARY:Consumer Price Index for July 2026
END:VEVENT
BEGIN:VEVENT
DTSTART;TZID=America/New_York:20260813T083000
SUMMARY:Producer Price Index for July 2026
END:VEVENT
END:VCALENDAR
"""

    events = MacroRiskController.parse_bls_ics(
        content,
        keywords=("Employment Situation", "Consumer Price Index"),
        pre_minutes=60,
        post_minutes=30,
    )

    assert [event.name for event in events] == ["US Nonfarm Payrolls / Employment Situation", "US CPI"]
    assert all(event.source == "BLS official calendar" for event in events)


def test_closed_one_minute_range_and_volume_shock_starts_cooldown() -> None:
    now_ms = 1_800_000
    baseline = [
        Candle(index * 60_000, 100.0, 101.0, 100.0, 100.5, 10.0)
        for index in range(20)
    ]
    shock = Candle(1_200_000, 100.0, 103.0, 99.5, 102.5, 40.0)
    config = MacroRiskConfig(
        enabled=True,
        bls_ics_url="",
        shock_lookback=20,
        shock_range_multiple=2.5,
        shock_volume_multiple=3.0,
        shock_extreme_range_multiple=4.0,
        shock_cooldown_minutes=30,
        shock_max_candle_age_minutes=10,
    )
    controller = MacroRiskController(config)

    decision = controller.decision({"1m": baseline + [shock]}, now_ms=now_ms)

    assert decision.blocked is True
    assert decision.reason.startswith("macro_shock:")
    assert decision.shock_until_ms == now_ms + 30 * 60_000


def test_repeated_shock_does_not_extend_active_cooldown() -> None:
    baseline = [
        Candle(index * 60_000, 100.0, 101.0, 100.0, 100.5, 10.0)
        for index in range(20)
    ]
    first_now_ms = 1_800_000
    first_shock = Candle(1_200_000, 100.0, 103.0, 99.5, 102.5, 40.0)
    second_shock = Candle(1_260_000, 102.5, 106.0, 102.0, 105.5, 50.0)
    controller = MacroRiskController(
        MacroRiskConfig(
            enabled=True,
            bls_ics_url="",
            shock_lookback=20,
            shock_range_multiple=2.5,
            shock_volume_multiple=3.0,
            shock_extreme_range_multiple=4.0,
            shock_cooldown_minutes=30,
            shock_max_candle_age_minutes=10,
        )
    )

    first = controller.decision({"1m": baseline + [first_shock]}, now_ms=first_now_ms)
    second = controller.decision(
        {"1m": baseline[1:] + [first_shock, second_shock]},
        now_ms=first_now_ms + 60_000,
    )

    assert second.blocked is True
    assert second.shock_until_ms == first.shock_until_ms


def test_directional_impulse_remains_tradable() -> None:
    now_ms = 1_800_000
    baseline = [
        Candle(index * 60_000, 100.0, 101.0, 100.0, 100.5, 10.0)
        for index in range(20)
    ]
    impulse = Candle(1_200_000, 100.0, 104.0, 99.5, 103.8, 40.0)
    controller = MacroRiskController(
        MacroRiskConfig(
            enabled=True,
            bls_ics_url="",
            shock_entry_policy="directional",
            shock_cooldown_minutes=3,
            shock_max_candle_age_minutes=10,
        )
    )

    decision = controller.decision({"1m": baseline + [impulse]}, now_ms=now_ms)

    assert decision.blocked is False
    assert decision.reason == ""
    assert decision.shock_classification == "tradable_directional"
    assert decision.shock_until_ms == 0


def test_directional_policy_only_blocks_disorderly_dislocation_briefly() -> None:
    now_ms = 1_800_000
    baseline = [
        Candle(index * 60_000, 100.0, 101.0, 100.0, 100.5, 10.0)
        for index in range(20)
    ]
    dislocation = Candle(1_200_000, 100.0, 107.0, 93.0, 100.5, 40.0)
    controller = MacroRiskController(
        MacroRiskConfig(
            enabled=True,
            bls_ics_url="",
            shock_entry_policy="directional",
            shock_cooldown_minutes=3,
            shock_max_candle_age_minutes=10,
        )
    )

    decision = controller.decision({"1m": baseline + [dislocation]}, now_ms=now_ms)

    assert decision.blocked is True
    assert decision.reason == "market_dislocation:range=14.00x,volume=4.00x"
    assert decision.shock_classification == "extreme_dislocation"
    assert decision.shock_until_ms == now_ms + 3 * 60_000
    assert decision.shock_trigger_range_ratio == 14.0
    assert decision.shock_trigger_volume_ratio == 4.0


def test_tiny_absolute_range_is_not_misclassified_as_extreme_dislocation() -> None:
    now_ms = 1_800_000
    baseline = [
        Candle(index * 60_000, 100.0, 100.001, 100.0, 100.001, 10.0)
        for index in range(20)
    ]
    relative_spike = Candle(1_200_000, 100.0, 100.02, 100.0, 100.001, 40.0)
    controller = MacroRiskController(
        MacroRiskConfig(
            enabled=True,
            bls_ics_url="",
            shock_entry_policy="directional",
            shock_cooldown_minutes=3,
            shock_max_candle_age_minutes=10,
            shock_dislocation_min_range_pct=0.001,
        )
    )

    decision = controller.decision({"1m": baseline + [relative_spike]}, now_ms=now_ms)

    assert decision.range_ratio > 12.0
    assert decision.blocked is False
    assert decision.reason == ""
    assert decision.shock_classification == "tradable_elevated_volatility"
    assert decision.shock_until_ms == 0


def test_dislocation_reason_retains_trigger_ratios_during_cooldown() -> None:
    now_ms = 1_800_000
    baseline = [
        Candle(index * 60_000, 100.0, 101.0, 100.0, 100.5, 10.0)
        for index in range(20)
    ]
    dislocation = Candle(1_200_000, 100.0, 107.0, 93.0, 100.5, 40.0)
    calm = Candle(1_260_000, 100.5, 101.5, 100.5, 101.0, 10.0)
    controller = MacroRiskController(
        MacroRiskConfig(
            enabled=True,
            bls_ics_url="",
            shock_entry_policy="directional",
            shock_cooldown_minutes=3,
            shock_max_candle_age_minutes=10,
        )
    )

    controller.decision({"1m": baseline + [dislocation]}, now_ms=now_ms)
    decision = controller.decision(
        {"1m": baseline[1:] + [dislocation, calm]},
        now_ms=now_ms + 60_000,
    )

    assert decision.blocked is True
    assert decision.range_ratio == 1.0
    assert decision.volume_ratio == 1.0
    assert decision.reason == "market_dislocation:range=14.00x,volume=4.00x"
    assert decision.shock_classification == "extreme_dislocation"


def test_engine_does_not_open_position_when_macro_window_is_blocked() -> None:
    class Adapter:
        name = "test"
        settings = SimpleNamespace(symbol="BTC-USDT")

        @staticmethod
        def fetch_candles(interval: str, limit: int) -> list[Candle]:
            price = {"1m": 105.0, "5m": 104.0, "1h": 103.0}[interval]
            step = {"1m": 60_000, "5m": 300_000, "1h": 3_600_000}[interval]
            return [
                Candle(0, price, price + 0.1, price - 0.1, price, 10.0),
                Candle(step, price, price + 0.1, price - 0.1, price, 10.0),
            ]

    class Strategy:
        config = StrategyConfig(trigger_timeframe="5m", regime_timeframe="1h")

        @staticmethod
        def evaluate(candles_by_timeframe: object) -> Signal:
            return Signal("long", 6, 1, ("fixed",))

    class BlockingMacro:
        config = MacroRiskConfig(enabled=True)

        @staticmethod
        def decision(candles_by_timeframe: object) -> MacroRiskDecision:
            return MacroRiskDecision(True, "macro_event:FOMC")

        @staticmethod
        def status() -> dict[str, object]:
            return {"enabled": True, "blocked": True, "reason": "macro_event:FOMC"}

    engine = TradingEngine(
        Adapter(),
        Strategy(),
        RiskManager(),
        EngineConfig(mode="paper"),
        macro_risk=BlockingMacro(),
    )

    result = engine.evaluate_once()

    assert result.status == "macro_event_blocked"
    assert result.position is None
    assert result.signal is not None
    assert "entry_blocked=macro_event:FOMC" in result.signal.reasons


def test_macro_blocked_signal_is_retried_after_window_clears() -> None:
    class Adapter:
        name = "test"
        settings = SimpleNamespace(symbol="BTC-USDT")

        @staticmethod
        def fetch_candles(interval: str, limit: int) -> list[Candle]:
            price = {"1m": 105.0, "5m": 104.0, "1h": 103.0}[interval]
            step = {"1m": 60_000, "5m": 300_000, "1h": 3_600_000}[interval]
            return [
                Candle(0, price, price + 0.1, price - 0.1, price, 10.0),
                Candle(step, price, price + 0.1, price - 0.1, price, 10.0),
            ]

    class Strategy:
        config = StrategyConfig(trigger_timeframe="5m", regime_timeframe="1h")
        calls = 0

        def evaluate(self, candles_by_timeframe: object) -> Signal:
            self.calls += 1
            if self.calls == 1:
                return Signal("long", 6, 1, ("fixed",))
            return Signal("flat", 5, 2, ("no_current_setup",))

        @staticmethod
        def reevaluate_blocked_signal(side: str, candles_by_timeframe: object) -> Signal:
            return Signal(side, 6, 2, ("fixed", "macro_blocked_signal_revalidated"))

    class MutableMacro:
        config = MacroRiskConfig(enabled=True)
        blocked = True

        def decision(self, candles_by_timeframe: object) -> MacroRiskDecision:
            return MacroRiskDecision(self.blocked, "macro_event:FOMC" if self.blocked else "")

        def status(self) -> dict[str, object]:
            return {"enabled": True, "blocked": self.blocked}

    macro = MutableMacro()
    engine = TradingEngine(
        Adapter(),
        Strategy(),
        RiskManager(),
        EngineConfig(mode="paper"),
        macro_risk=macro,
    )

    blocked = engine.evaluate_once()
    macro.blocked = False
    retried = engine.evaluate_once()

    assert blocked.status == "macro_event_blocked"
    assert engine.last_signal_timestamp == 2
    assert retried.status == "paper_signal"
    assert retried.position is not None
    assert retried.signal is not None
    assert "macro_blocked_signal_revalidated" in retried.signal.reasons


def test_weak_position_closes_before_major_event() -> None:
    event = MacroEvent("FOMC", int(time.time() * 1000) + 10 * 60_000, pre_minutes=240, post_minutes=120)
    controller = MacroRiskController(
        MacroRiskConfig(enabled=True, bls_ics_url="", position_min_r=0.5, events=(event,))
    )
    engine = TradingEngine(
        SimpleNamespace(name="test"),
        SimpleNamespace(config=StrategyConfig()),
        RiskManager(),
        EngineConfig(mode="paper"),
        macro_risk=controller,
    )
    engine.position = Position(
        "long",
        1.0,
        100.0,
        99.0,
        102.5,
        int(time.time() * 1000) - 60_000,
        initial_stop_price=99.0,
    )
    decision = MacroRiskDecision(True, "macro_event:FOMC", event, seconds_to_event=600)

    engine._manage_macro_position(Candle(1, 100.0, 100.3, 99.9, 100.2, 10.0), decision)

    assert engine.position is None


def test_profitable_position_moves_to_cost_break_even_before_event() -> None:
    event = MacroEvent("FOMC", int(time.time() * 1000) + 10 * 60_000, pre_minutes=240, post_minutes=120)
    controller = MacroRiskController(
        MacroRiskConfig(enabled=True, bls_ics_url="", position_min_r=0.5, events=(event,))
    )
    engine = TradingEngine(
        SimpleNamespace(name="test"),
        SimpleNamespace(config=StrategyConfig()),
        RiskManager(),
        EngineConfig(mode="paper"),
        macro_risk=controller,
    )
    engine.position = Position(
        "long",
        1.0,
        100.0,
        99.0,
        102.5,
        int(time.time() * 1000) - 60_000,
        initial_stop_price=99.0,
    )
    decision = MacroRiskDecision(True, "macro_event:FOMC", event, seconds_to_event=600)

    engine._manage_macro_position(Candle(1, 100.4, 100.7, 100.3, 100.6, 10.0), decision)

    assert engine.position is not None
    assert engine.position.stop_reason == "macro_event_stop"
    assert engine.position.stop_price >= engine.risk.break_even_price("long", 100.0)
