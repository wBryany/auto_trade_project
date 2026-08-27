from __future__ import annotations

import json
import logging
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

from .models import Candle


LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class MacroEvent:
    name: str
    timestamp_ms: int
    pre_minutes: int = 60
    post_minutes: int = 30
    source: str = "config"

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], default_timezone: str = "America/New_York") -> "MacroEvent":
        value = str(raw.get("at") or raw.get("time") or "").strip()
        if not value:
            raise ValueError("macro event requires an 'at' timestamp")
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=ZoneInfo(str(raw.get("timezone") or default_timezone)))
        return cls(
            name=str(raw.get("name") or "macro_event").strip(),
            timestamp_ms=int(parsed.timestamp() * 1000),
            pre_minutes=max(0, int(raw.get("pre_minutes", 60))),
            post_minutes=max(0, int(raw.get("post_minutes", 30))),
            source=str(raw.get("source") or "config"),
        )


@dataclass(frozen=True)
class MacroRiskDecision:
    blocked: bool = False
    reason: str = ""
    event: MacroEvent | None = None
    seconds_to_event: float | None = None
    shock_until_ms: int = 0
    range_ratio: float = 0.0
    volume_ratio: float = 0.0
    shock_classification: str = ""
    shock_trigger_range_ratio: float = 0.0
    shock_trigger_volume_ratio: float = 0.0
    shock_triggered_at_ms: int = 0


@dataclass(frozen=True)
class MacroRiskConfig:
    enabled: bool = False
    cache_path: str = "reports/macro_events_cache.json"
    refresh_seconds: int = 21_600
    fetch_timeout_seconds: float = 4.0
    bls_ics_url: str = "https://www.bls.gov/schedule/news_release/bls.ics"
    bls_keywords: tuple[str, ...] = ("Employment Situation", "Consumer Price Index")
    bls_pre_minutes: int = 60
    bls_post_minutes: int = 30
    position_management_before_minutes: int = 30
    position_min_r: float = 0.5
    shock_lookback: int = 20
    shock_range_multiple: float = 2.5
    shock_volume_multiple: float = 3.0
    shock_extreme_range_multiple: float = 4.0
    shock_cooldown_minutes: int = 30
    shock_max_candle_age_minutes: int = 5
    shock_entry_policy: str = "hard_block"
    shock_directional_min_body_ratio: float = 0.55
    shock_directional_min_close_location: float = 0.70
    shock_dislocation_range_multiple: float = 6.0
    shock_dislocation_min_range_pct: float = 0.0
    events: tuple[MacroEvent, ...] = ()

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None, *, default_cache_path: str = "") -> "MacroRiskConfig":
        raw = raw or {}
        default_timezone = str(raw.get("default_timezone") or "America/New_York")
        events: list[MacroEvent] = []
        for item in raw.get("events", ()):
            try:
                events.append(MacroEvent.from_mapping(item, default_timezone))
            except (TypeError, ValueError, KeyError) as error:
                LOG.warning("ignoring invalid macro event %r: %s", item, error)
        keywords = tuple(str(item).strip() for item in raw.get("bls_keywords", cls.bls_keywords) if str(item).strip())
        shock_entry_policy = str(raw.get("shock_entry_policy") or cls.shock_entry_policy).strip().lower()
        if shock_entry_policy not in {"hard_block", "directional"}:
            LOG.warning("invalid shock_entry_policy %r; using hard_block", shock_entry_policy)
            shock_entry_policy = "hard_block"
        return cls(
            enabled=bool(raw.get("enabled", False)),
            cache_path=str(raw.get("cache_path") or default_cache_path or cls.cache_path),
            refresh_seconds=max(300, int(raw.get("refresh_seconds", cls.refresh_seconds))),
            fetch_timeout_seconds=max(1.0, float(raw.get("fetch_timeout_seconds", cls.fetch_timeout_seconds))),
            bls_ics_url=str(raw.get("bls_ics_url") or cls.bls_ics_url),
            bls_keywords=keywords,
            bls_pre_minutes=max(0, int(raw.get("bls_pre_minutes", cls.bls_pre_minutes))),
            bls_post_minutes=max(0, int(raw.get("bls_post_minutes", cls.bls_post_minutes))),
            position_management_before_minutes=max(
                0,
                int(raw.get("position_management_before_minutes", cls.position_management_before_minutes)),
            ),
            position_min_r=max(0.0, float(raw.get("position_min_r", cls.position_min_r))),
            shock_lookback=max(5, int(raw.get("shock_lookback", cls.shock_lookback))),
            shock_range_multiple=max(1.0, float(raw.get("shock_range_multiple", cls.shock_range_multiple))),
            shock_volume_multiple=max(1.0, float(raw.get("shock_volume_multiple", cls.shock_volume_multiple))),
            shock_extreme_range_multiple=max(
                1.0,
                float(raw.get("shock_extreme_range_multiple", cls.shock_extreme_range_multiple)),
            ),
            shock_cooldown_minutes=max(1, int(raw.get("shock_cooldown_minutes", cls.shock_cooldown_minutes))),
            shock_max_candle_age_minutes=max(
                1,
                int(raw.get("shock_max_candle_age_minutes", cls.shock_max_candle_age_minutes)),
            ),
            shock_entry_policy=shock_entry_policy,
            shock_directional_min_body_ratio=min(
                1.0,
                max(
                    0.0,
                    float(
                        raw.get(
                            "shock_directional_min_body_ratio",
                            cls.shock_directional_min_body_ratio,
                        )
                    ),
                ),
            ),
            shock_directional_min_close_location=min(
                1.0,
                max(
                    0.5,
                    float(
                        raw.get(
                            "shock_directional_min_close_location",
                            cls.shock_directional_min_close_location,
                        )
                    ),
                ),
            ),
            shock_dislocation_range_multiple=max(
                1.0,
                float(
                    raw.get(
                        "shock_dislocation_range_multiple",
                        cls.shock_dislocation_range_multiple,
                    )
                ),
            ),
            shock_dislocation_min_range_pct=min(
                1.0,
                max(
                    0.0,
                    float(
                        raw.get(
                            "shock_dislocation_min_range_pct",
                            cls.shock_dislocation_min_range_pct,
                        )
                    ),
                ),
            ),
            events=tuple(events),
        )


class MacroRiskController:
    """Block entries around high-impact releases and sudden market shocks.

    Scheduled releases come from explicit official-calendar entries plus the
    BLS iCalendar feed.  Remote data is cached and all refresh failures are
    contained here so macro risk can never terminate the market-data loop.
    """

    def __init__(self, config: MacroRiskConfig, *, now_fn: Callable[[], float] = time.time) -> None:
        self.config = config
        self._now_fn = now_fn
        self._configured_events = list(config.events)
        self._remote_events: list[MacroEvent] = []
        self._last_refresh_attempt_ms = 0
        self._last_refresh_error = ""
        self._last_shock_candle_timestamp = 0
        self._shock_until_ms = 0
        self._last_shock_classification = ""
        self._shock_trigger_classification = ""
        self._shock_trigger_range_ratio = 0.0
        self._shock_trigger_volume_ratio = 0.0
        self._shock_triggered_at_ms = 0
        self._last_decision = MacroRiskDecision()
        self._load_cache()

    def decision(
        self,
        candles_by_timeframe: Mapping[str, Sequence[Candle]],
        *,
        now_ms: int | None = None,
    ) -> MacroRiskDecision:
        if not self.config.enabled:
            self._last_decision = MacroRiskDecision()
            return self._last_decision
        selected_now = int(self._now_fn() * 1000) if now_ms is None else int(now_ms)
        self._refresh_if_due(selected_now)
        range_ratio, volume_ratio = self._detect_shock(candles_by_timeframe.get("1m", ()), selected_now)

        events = self._events()
        active = [
            event
            for event in events
            if event.timestamp_ms - event.pre_minutes * 60_000
            <= selected_now
            <= event.timestamp_ms + event.post_minutes * 60_000
        ]
        active_event = min(active, key=lambda item: abs(item.timestamp_ms - selected_now)) if active else None
        upcoming = [event for event in events if event.timestamp_ms >= selected_now]
        next_event = min(upcoming, key=lambda item: item.timestamp_ms) if upcoming else None
        display_event = active_event or next_event
        seconds_to_event = (display_event.timestamp_ms - selected_now) / 1000 if display_event else None

        shock_blocked = selected_now < self._shock_until_ms
        if active_event is not None:
            reason = f"macro_event:{active_event.name}"
        elif shock_blocked:
            reason_prefix = (
                "macro_shock"
                if self._shock_trigger_classification == "legacy_hard_block"
                else "market_dislocation"
            )
            reason = (
                f"{reason_prefix}:"
                f"range={self._shock_trigger_range_ratio:.2f}x,"
                f"volume={self._shock_trigger_volume_ratio:.2f}x"
            )
        else:
            reason = ""
        self._last_decision = MacroRiskDecision(
            blocked=bool(active_event or shock_blocked),
            reason=reason,
            event=display_event,
            seconds_to_event=seconds_to_event,
            shock_until_ms=self._shock_until_ms,
            range_ratio=range_ratio,
            volume_ratio=volume_ratio,
            shock_classification=(
                self._shock_trigger_classification if shock_blocked else self._last_shock_classification
            ),
            shock_trigger_range_ratio=self._shock_trigger_range_ratio,
            shock_trigger_volume_ratio=self._shock_trigger_volume_ratio,
            shock_triggered_at_ms=self._shock_triggered_at_ms,
        )
        return self._last_decision

    def status(self) -> dict[str, Any]:
        decision = self._last_decision
        event = decision.event
        return {
            "enabled": self.config.enabled,
            "shock_entry_policy": self.config.shock_entry_policy,
            "shock_cooldown_minutes": self.config.shock_cooldown_minutes,
            "blocked": decision.blocked,
            "reason": decision.reason,
            "next_event": (
                {
                    "name": event.name,
                    "timestamp_ms": event.timestamp_ms,
                    "pre_minutes": event.pre_minutes,
                    "post_minutes": event.post_minutes,
                    "source": event.source,
                    "seconds_to_event": decision.seconds_to_event,
                }
                if event
                else None
            ),
            "shock_until_ms": decision.shock_until_ms,
            "range_ratio": decision.range_ratio,
            "volume_ratio": decision.volume_ratio,
            "shock_classification": decision.shock_classification,
            "shock_trigger_range_ratio": decision.shock_trigger_range_ratio,
            "shock_trigger_volume_ratio": decision.shock_trigger_volume_ratio,
            "shock_triggered_at_ms": decision.shock_triggered_at_ms,
            "calendar_events": len(self._events()),
            "calendar_error": self._last_refresh_error,
        }

    def _events(self) -> list[MacroEvent]:
        unique: dict[int, MacroEvent] = {}
        # Explicitly configured official releases override matching cached/feed
        # rows so a fallback entry cannot create a duplicate blackout window.
        for event in self._remote_events + self._configured_events:
            unique[event.timestamp_ms] = event
        return sorted(unique.values(), key=lambda item: item.timestamp_ms)

    def _detect_shock(self, candles: Sequence[Candle], now_ms: int) -> tuple[float, float]:
        needed = self.config.shock_lookback + 1
        if len(candles) < needed:
            return 0.0, 0.0
        current = candles[-1]
        if current.timestamp <= self._last_shock_candle_timestamp:
            return self._last_decision.range_ratio, self._last_decision.volume_ratio
        self._last_shock_candle_timestamp = current.timestamp
        self._last_shock_classification = ""
        candle_age = now_ms - (int(current.timestamp) + 60_000)
        if candle_age < -60_000 or candle_age > self.config.shock_max_candle_age_minutes * 60_000:
            return 0.0, 0.0

        baseline = candles[-needed:-1]
        ranges = [max(0.0, float(item.high) - float(item.low)) for item in baseline]
        current_range = max(0.0, float(current.high) - float(current.low))
        median_range = statistics.median(ranges) if any(ranges) else 0.0
        range_ratio = current_range / median_range if median_range > 0 else 0.0

        def selected_volume(item: Candle) -> float:
            return float(item.quote_volume if item.quote_volume is not None else item.volume)

        volumes = [max(0.0, selected_volume(item)) for item in baseline]
        current_volume = max(0.0, selected_volume(current))
        median_volume = statistics.median(volumes) if any(volumes) else 0.0
        volume_ratio = current_volume / median_volume if median_volume > 0 else 0.0
        ordinary_shock = (
            range_ratio >= self.config.shock_range_multiple
            and volume_ratio >= self.config.shock_volume_multiple
        )
        extreme_range = range_ratio >= self.config.shock_extreme_range_multiple
        if ordinary_shock or extreme_range:
            reference_price = max(abs(float(current.open)), abs(float(current.close)))
            absolute_range_pct = current_range / reference_price if reference_price > 0 else 0.0
            absolute_dislocation = absolute_range_pct >= self.config.shock_dislocation_min_range_pct
            body_ratio = abs(float(current.close) - float(current.open)) / current_range if current_range else 0.0
            if float(current.close) >= float(current.open):
                close_location = (float(current.close) - float(current.low)) / current_range if current_range else 0.5
            else:
                close_location = (float(current.high) - float(current.close)) / current_range if current_range else 0.5
            directional = (
                body_ratio >= self.config.shock_directional_min_body_ratio
                and close_location >= self.config.shock_directional_min_close_location
            )
            disorderly = (
                absolute_dislocation
                and range_ratio >= self.config.shock_dislocation_range_multiple
                and not directional
            )
            extreme_dislocation = (
                absolute_dislocation
                and range_ratio >= self.config.shock_dislocation_range_multiple * 2.0
            )
            if self.config.shock_entry_policy == "directional":
                should_block = disorderly or extreme_dislocation
                if should_block:
                    classification = "extreme_dislocation" if extreme_dislocation else "disorderly_dislocation"
                elif directional:
                    classification = "tradable_directional"
                else:
                    classification = "tradable_elevated_volatility"
            else:
                should_block = True
                classification = "legacy_hard_block"
            self._last_shock_classification = classification

            if should_block and now_ms >= self._shock_until_ms:
                self._shock_until_ms = now_ms + self.config.shock_cooldown_minutes * 60_000
                self._shock_trigger_classification = classification
                self._shock_trigger_range_ratio = range_ratio
                self._shock_trigger_volume_ratio = volume_ratio
                self._shock_triggered_at_ms = now_ms
                LOG.warning(
                    "market dislocation circuit breaker class=%s range=%.2fx volume=%.2fx cooldown=%sm",
                    classification,
                    range_ratio,
                    volume_ratio,
                    self.config.shock_cooldown_minutes,
                )
            elif should_block:
                LOG.info(
                    "market dislocation observed during active cooldown range=%.2fx volume=%.2fx; cooldown not extended",
                    range_ratio,
                    volume_ratio,
                )
            else:
                LOG.info(
                    "tradable volatility class=%s range=%.2fx volume=%.2fx body=%.2f close_location=%.2f",
                    classification,
                    range_ratio,
                    volume_ratio,
                    body_ratio,
                    close_location,
                )
        return range_ratio, volume_ratio

    def _refresh_if_due(self, now_ms: int) -> None:
        if not self.config.bls_ics_url:
            return
        if now_ms - self._last_refresh_attempt_ms < self.config.refresh_seconds * 1000:
            return
        self._last_refresh_attempt_ms = now_ms
        try:
            request = Request(
                self.config.bls_ics_url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/138 Safari/537.36",
                    "Accept": "text/calendar,text/plain;q=0.9,*/*;q=0.8",
                    "Referer": "https://www.bls.gov/schedule/",
                },
            )
            with urlopen(request, timeout=self.config.fetch_timeout_seconds) as response:
                content = response.read().decode("utf-8-sig", errors="replace")
            self._remote_events = self.parse_bls_ics(
                content,
                keywords=self.config.bls_keywords,
                pre_minutes=self.config.bls_pre_minutes,
                post_minutes=self.config.bls_post_minutes,
            )
            self._last_refresh_error = ""
            self._save_cache()
        except Exception as error:  # Network/calendar errors must not stop trading cycles.
            self._last_refresh_error = str(error)
            LOG.warning("macro calendar refresh failed; using cached/configured events: %s", error)

    @staticmethod
    def parse_bls_ics(
        content: str,
        *,
        keywords: Sequence[str],
        pre_minutes: int,
        post_minutes: int,
    ) -> list[MacroEvent]:
        unfolded: list[str] = []
        for raw_line in content.replace("\r\n", "\n").split("\n"):
            if raw_line.startswith((" ", "\t")) and unfolded:
                unfolded[-1] += raw_line[1:]
            else:
                unfolded.append(raw_line.strip("\r"))
        events: list[MacroEvent] = []
        block: list[str] = []
        in_event = False
        for line in unfolded:
            if line == "BEGIN:VEVENT":
                block = []
                in_event = True
                continue
            if line == "END:VEVENT":
                if in_event:
                    event = MacroRiskController._parse_bls_event(
                        block,
                        keywords=keywords,
                        pre_minutes=pre_minutes,
                        post_minutes=post_minutes,
                    )
                    if event:
                        events.append(event)
                in_event = False
                continue
            if in_event:
                block.append(line)
        return sorted(events, key=lambda item: item.timestamp_ms)

    @staticmethod
    def _parse_bls_event(
        lines: Sequence[str],
        *,
        keywords: Sequence[str],
        pre_minutes: int,
        post_minutes: int,
    ) -> MacroEvent | None:
        summary = ""
        dtstart_key = ""
        dtstart_value = ""
        for line in lines:
            if line.startswith("SUMMARY:"):
                summary = line.split(":", 1)[1].replace("\\,", ",").replace("\\n", " ").strip()
            elif line.startswith("DTSTART") and ":" in line:
                dtstart_key, dtstart_value = line.split(":", 1)
        if not summary or not any(keyword.casefold() in summary.casefold() for keyword in keywords):
            return None
        if not dtstart_value or "T" not in dtstart_value:
            return None
        timezone_name = "America/New_York"
        if "TZID=" in dtstart_key:
            timezone_name = dtstart_key.split("TZID=", 1)[1].split(";", 1)[0]
            if timezone_name in {"Eastern Standard Time", "US-Eastern"}:
                timezone_name = "America/New_York"
        value = dtstart_value.strip()
        selected_timezone = timezone.utc if value.endswith("Z") else ZoneInfo(timezone_name)
        value = value.removesuffix("Z")
        parsed = None
        for pattern in ("%Y%m%dT%H%M%S", "%Y%m%dT%H%M"):
            try:
                parsed = datetime.strptime(value, pattern).replace(tzinfo=selected_timezone)
                break
            except ValueError:
                continue
        if parsed is None:
            return None
        if "employment situation" in summary.casefold():
            name = "US Nonfarm Payrolls / Employment Situation"
        elif "consumer price index" in summary.casefold():
            name = "US CPI"
        else:
            name = summary
        return MacroEvent(
            name=name,
            timestamp_ms=int(parsed.timestamp() * 1000),
            pre_minutes=pre_minutes,
            post_minutes=post_minutes,
            source="BLS official calendar",
        )

    def _load_cache(self) -> None:
        path = Path(self.config.cache_path)
        if not path.exists():
            return
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            self._remote_events = [
                MacroEvent(
                    name=str(item["name"]),
                    timestamp_ms=int(item["timestamp_ms"]),
                    pre_minutes=int(item.get("pre_minutes", self.config.bls_pre_minutes)),
                    post_minutes=int(item.get("post_minutes", self.config.bls_post_minutes)),
                    source=str(item.get("source") or "cached calendar"),
                )
                for item in raw.get("events", ())
            ]
        except Exception as error:
            LOG.warning("macro calendar cache ignored: %s", error)

    def _save_cache(self) -> None:
        path = Path(self.config.cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "events": [
                {
                    "name": event.name,
                    "timestamp_ms": event.timestamp_ms,
                    "pre_minutes": event.pre_minutes,
                    "post_minutes": event.post_minutes,
                    "source": event.source,
                }
                for event in self._remote_events
            ],
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
