from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

from .engine import EngineConfig, TradingEngine, normalized_poll_seconds
from .costs import CostConfig
from .exchanges.factory import make_adapter
from .macro_risk import MacroRiskConfig, MacroRiskController
from .notifications import EmailNotificationConfig, EmailNotifier
from .risk import RiskConfig, RiskManager
from .reporting import TradeReporter
from .strategy import MultiTimeframeStrategy, StrategyConfig


EXCHANGE_DEFAULTS: dict[str, dict[str, Any]] = {
    "okx": {
        "environment": "demo",
        "base_url": "https://openapi.okx.com",
        "symbol": "BTC-USDT-SWAP",
        "contract_size": 0,
        "api_key_env": "OKX_API_KEY",
        "api_secret_env": "OKX_API_SECRET",
        "passphrase_env": "OKX_API_PASSPHRASE",
    },
    "binance": {
        "environment": "testnet",
        "base_url": "https://demo-fapi.binance.com",
        "symbol": "BTCUSDT",
        "contract_size": 1,
        "api_key_env": "BINANCE_TESTNET_API_KEY",
        "api_secret_env": "BINANCE_TESTNET_API_SECRET",
    },
    "gate": {
        "environment": "testnet",
        "base_url": "https://api-testnet.gateapi.io/api/v4",
        "settle": "usdt",
        "symbol": "BTC_USDT",
        "contract_size": 0,
        "api_key_env": "GATE_API_KEY",
        "api_secret_env": "GATE_API_SECRET",
    },
}


BINANCE_ENVIRONMENTS = {
    "testnet": "https://demo-fapi.binance.com",
    "production": "https://fapi.binance.com",
}
BINANCE_CREDENTIAL_ENVS = {
    "testnet": ("BINANCE_TESTNET_API_KEY", "BINANCE_TESTNET_API_SECRET"),
    "production": ("BINANCE_LIVE_API_KEY", "BINANCE_LIVE_API_SECRET"),
}
_CREDENTIAL_FIELDS = ("api_key", "api_secret", "passphrase")


def canonical_binance_environment(value: Any) -> str:
    environment = str(value or "testnet").strip().lower()
    aliases = {"demo": "testnet", "live": "production", "mainnet": "production", "prod": "production"}
    environment = aliases.get(environment, environment)
    if environment not in BINANCE_ENVIRONMENTS:
        raise ValueError("Binance environment must be testnet or production")
    return environment


def normalize_binance_symbol(value: Any) -> str:
    symbol = str(value or "BTCUSDT").strip().upper()
    aliases = {
        "BTC-USDT-SWAP": "BTCUSDT",
        "BTC-USDT": "BTCUSDT",
        "BTC_USDT": "BTCUSDT",
    }
    symbol = aliases.get(symbol, symbol)
    if not symbol or not symbol.isascii() or not symbol.isalnum():
        raise ValueError("Binance USD-M symbol must use exchange format, for example BTCUSDT")
    return symbol


def credential_values(config: dict[str, Any], name: str) -> dict[str, Any]:
    """Return credentials for exactly one venue environment.

    Legacy flat Binance credentials are accepted only for testnet. Production
    never falls back to them, which prevents a test key (or the wrong account)
    from being reused after an environment switch.
    """
    saved = config.get("credentials", {}).get(name, {})
    if not isinstance(saved, dict):
        return {}
    if name != "binance":
        return saved
    environment = canonical_binance_environment(
        config.get("exchanges", {}).get("binance", {}).get("environment", "testnet")
    )
    scoped = saved.get(environment, {})
    if isinstance(scoped, dict) and any(str(scoped.get(key, "")).strip() for key in _CREDENTIAL_FIELDS):
        return scoped
    if environment == "testnet":
        return {key: saved.get(key, "") for key in _CREDENTIAL_FIELDS}
    return {}


def report_directory(config: dict[str, Any], exchange_name: str) -> Path:
    base = Path(config.get("report_dir", "reports"))
    exchange = config.get("exchanges", {}).get(exchange_name, {})
    if (
        exchange_name == "binance"
        and canonical_binance_environment(exchange.get("environment", "testnet")) == "production"
        and str(config.get("mode", "paper")) == "live"
    ):
        return base / "binance-production"
    return base


def ensure_exchange_defaults(config: dict[str, Any], name: str) -> dict[str, Any]:
    exchanges = config.setdefault("exchanges", {})
    exchange = exchanges.setdefault(name, {})
    for key, value in EXCHANGE_DEFAULTS.get(name, {}).items():
        if key not in exchange or exchange.get(key) in {None, ""}:
            exchange[key] = value
    if name == "okx" and exchange.get("base_url", "").rstrip("/") == "https://www.okx.com":
        exchange["base_url"] = "https://openapi.okx.com"
    # The first page version used Binance values when the user switched platforms.
    # Correct only those known cross-platform defaults; preserve deliberate symbols.
    if name == "okx" and exchange.get("symbol") in {"", "BTCUSDT"}:
        exchange["symbol"] = "BTC-USDT-SWAP"
    if name == "okx" and exchange.get("environment") == "testnet":
        exchange["environment"] = "demo"
    if name == "gate" and exchange.get("symbol") in {"", "BTCUSDT"}:
        exchange["symbol"] = "BTC_USDT"
    if name == "binance":
        environment = canonical_binance_environment(exchange.get("environment", "testnet"))
        exchange["environment"] = environment
        # Endpoint selection is intentionally server-side. Never accept a
        # dashboard-supplied host for a credential-bearing Binance request.
        exchange["base_url"] = BINANCE_ENVIRONMENTS[environment]
        exchange["symbol"] = normalize_binance_symbol(exchange.get("symbol", "BTCUSDT"))
        exchange["api_key_env"], exchange["api_secret_env"] = BINANCE_CREDENTIAL_ENVS[environment]
    return exchange


def local_config_path(path: str | Path) -> Path:
    """Return the ignored, credential-bearing companion config path."""
    config_path = Path(path)
    if config_path.name.endswith(".local.json"):
        return config_path
    return config_path.with_name(f"{config_path.stem}.local.json")


def _apply_config_credentials(config: dict[str, Any]) -> None:
    for name, exchange in config.get("exchanges", {}).items():
        selected_credentials = credential_values(config, name)
        for config_name, env_name in (("api_key", exchange.get("api_key_env")), ("api_secret", exchange.get("api_secret_env")), ("passphrase", exchange.get("passphrase_env"))):
            value = selected_credentials.get(config_name)
            if env_name and value:
                os.environ[str(env_name)] = str(value)


def load_config(path: str) -> dict[str, Any]:
    requested_path = Path(path)
    selected_path = local_config_path(requested_path)
    if not selected_path.exists():
        selected_path = requested_path
    with selected_path.open("r", encoding="utf-8") as file:
        config = json.load(file)
    config.setdefault("live_reconciliation_seconds", 5)
    config.setdefault("dashboard_snapshot_seconds", 15)
    config.setdefault("dashboard_private_stale_seconds", 90)
    config.setdefault("private_entry_retry_seconds", 45)
    config.setdefault("private_entry_retry_interval_seconds", 5)
    config.setdefault("candle_refresh_seconds", {"1m": 1, "5m": 3, "1h": 15})
    for name in config.get("exchanges", {}):
        ensure_exchange_defaults(config, name)
    strategy = config.setdefault("strategy", {})
    for key, default in (("mode", "scalp"), ("trigger_timeframe", "30s"), ("regime_timeframe", "5m"), ("max_hold_seconds", 420), ("hard_max_hold_seconds", 900), ("take_profit_r", 2.0), ("atr_stop_multiplier", 1.4), ("min_stop_loss_pct", 0.0025), ("max_stop_loss_pct", 0.006), ("structure_stop_lookback_bars", 0), ("structure_stop_buffer_atr", 0.0), ("min_hold_seconds", 90), ("reversal_min_score", 5), ("require_volume_confirmation", True)):
        strategy.setdefault(key, default)
    strategy.setdefault("volume_sma_period", 20)
    strategy.setdefault("min_volume_ratio", 1.3)
    strategy.setdefault("require_full_alignment", False)
    if strategy.get("require_full_alignment"):
        strategy["min_score"] = max(4, int(strategy.get("min_score", 4)))
    if not config.get("active_exchange"):
        for name in ("okx", "binance", "gate"):
            saved = credential_values(config, name)
            if any(str(saved.get(key, "")).strip() for key in ("api_key", "api_secret", "passphrase")):
                config["active_exchange"] = name
                break
    _apply_config_credentials(config)
    return config


def save_dashboard_config(path: str, updates: dict[str, Any]) -> Path:
    """Save dashboard settings and credentials to an ignored local config."""
    requested_path = Path(path)
    target = local_config_path(requested_path)
    source = target if target.exists() else requested_path
    with source.open("r", encoding="utf-8") as file:
        config = json.load(file)

    exchange_name = str(updates.get("exchange", "binance"))
    exchange = ensure_exchange_defaults(config, exchange_name)
    config["active_exchange"] = exchange_name
    if exchange_name == "binance" and "environment" in updates:
        exchange["environment"] = canonical_binance_environment(updates.get("environment"))
        exchange = ensure_exchange_defaults(config, exchange_name)
    credentials_root = config.setdefault("credentials", {}).setdefault(exchange_name, {})
    credentials = (
        credentials_root.setdefault(exchange["environment"], {})
        if exchange_name == "binance"
        else credentials_root
    )
    for key in ("api_key", "api_secret", "passphrase"):
        value = str(updates.get(key, "")).strip()
        if value:
            credentials[key] = value

    requested_mode = str(updates.get("mode", config.get("mode", "paper"))).strip().lower()
    if requested_mode not in {"paper", "live"}:
        raise ValueError("mode must be paper or live")
    config["mode"] = requested_mode
    if "poll_seconds" in updates:
        config["poll_seconds"] = normalized_poll_seconds(updates["poll_seconds"])
    if "paper_equity" in updates:
        config["paper_equity"] = float(updates["paper_equity"])
    account = config.setdefault("account", {})
    if "max_leverage" in updates:
        account["max_leverage"] = max(1, min(125, float(updates["max_leverage"])))
    if "symbol" in updates and str(updates["symbol"]).strip():
        exchange["symbol"] = (
            normalize_binance_symbol(updates["symbol"])
            if exchange_name == "binance"
            else str(updates["symbol"]).strip().upper()
        )
    risk = config.setdefault("risk", {})
    for key in ("stop_loss_pct", "risk_per_trade", "max_notional_pct"):
        if key in updates:
            risk[key] = float(updates[key])
    strategy = config.setdefault("strategy", {})
    # The dashboard's ``mode`` field is the execution mode (paper/live), not
    # the strategy selector (scalp/traditional_kline).  Keeping the names
    # separate prevents an ordinary dashboard save from silently disabling
    # the configured strategy implementation.
    if "strategy_mode" in updates and str(updates["strategy_mode"]).strip():
        strategy_mode = str(updates["strategy_mode"]).strip()
        if strategy_mode not in {"scalp", "traditional_kline"}:
            raise ValueError("strategy_mode must be scalp or traditional_kline")
        strategy["mode"] = strategy_mode
    for key in ("trigger_timeframe", "regime_timeframe"):
        if key in updates and str(updates[key]).strip():
            strategy[key] = str(updates[key]).strip()
    if "max_hold_seconds" in updates:
        strategy["max_hold_seconds"] = max(30, int(updates["max_hold_seconds"]))
    for key in ("min_score", "take_profit_r"):
        if key in updates:
            strategy[key] = float(updates[key]) if key == "take_profit_r" else int(updates[key])
    if "volume_sma_period" in updates:
        strategy["volume_sma_period"] = max(5, int(updates["volume_sma_period"]))
    if "min_volume_ratio" in updates:
        strategy["min_volume_ratio"] = max(1.0, float(updates["min_volume_ratio"]))
    if "require_full_alignment" in updates:
        strategy["require_full_alignment"] = bool(updates["require_full_alignment"])
    if "email_notifications" in updates:
        requested_email = updates.get("email_notifications") or {}
        report_dir = Path(config.get("report_dir", "reports"))
        validated_email = EmailNotificationConfig.from_mapping(
            requested_email,
            default_timezone=str(config.get("report_timezone", "Asia/Shanghai")),
            default_state_path=str(report_dir / "email_notification_state.json"),
        )
        config["email_notifications"] = {
            "enabled": validated_email.enabled,
            "smtp_host": validated_email.smtp_host,
            "smtp_port": validated_email.smtp_port,
            "security": validated_email.security,
            "sender": validated_email.sender,
            "recipients": list(validated_email.recipients),
            "username_env": validated_email.username_env,
            "password_env": validated_email.password_env,
            "timeout_seconds": validated_email.timeout_seconds,
            "daily_report_enabled": validated_email.daily_report_enabled,
            "daily_report_hour": validated_email.daily_report_hour,
            "timezone": validated_email.timezone,
            "state_path": validated_email.state_path,
            "retry_seconds": validated_email.retry_seconds,
        }

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=2)
        file.write("\n")
    os.replace(temporary, target)
    _apply_config_credentials(config)
    return target


def load_dotenv(path: str = ".env") -> None:
    """Load simple KEY=VALUE entries without adding a dependency."""
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value


def build_engine(name: str, raw: dict[str, Any], reporter: TradeReporter | None = None) -> TradingEngine:
    account = raw.get("account", {})
    strategy = MultiTimeframeStrategy(StrategyConfig(**raw.get("strategy", {})))
    exchange_raw = raw["exchanges"][name]
    cost_raw = exchange_raw.get("costs", raw.get("costs", {}))
    costs = CostConfig(**cost_raw)
    risk = RiskManager(RiskConfig(**raw.get("risk", {})), max_leverage=float(account.get("max_leverage", 3)), costs=costs)
    adapter = make_adapter(name, exchange_raw, account)
    selected_report_dir = report_directory(raw, name)
    engine_config = EngineConfig(
        mode=str(raw.get("mode", "paper")),
        poll_seconds=int(raw.get("poll_seconds", 15)),
        paper_equity=float(raw.get("paper_equity", 10000)),
        candle_limit=int(raw.get("candle_limit", 300)),
        take_profit_r=float(raw.get("strategy", {}).get("take_profit_r", 1.6)),
        reconciliation_state_path=str(selected_report_dir / "live_reconciliation_state.json"),
        live_reconciliation_seconds=float(raw.get("live_reconciliation_seconds", 5)),
        private_entry_retry_seconds=float(raw.get("private_entry_retry_seconds", 45)),
        private_entry_retry_interval_seconds=float(
            raw.get("private_entry_retry_interval_seconds", 5)
        ),
        candle_refresh_seconds={
            str(timeframe): float(seconds)
            for timeframe, seconds in dict(
                raw.get("candle_refresh_seconds", {"1m": 1, "5m": 3, "1h": 15})
            ).items()
        },
    )
    if engine_config.mode not in {"paper", "live"}:
        raise ValueError("mode must be paper or live")
    if engine_config.mode == "live":
        logging.getLogger(__name__).warning("LIVE mode requested for %s; private order calls are enabled", name)
    default_macro_cache = str(selected_report_dir / "macro_events_cache.json")
    macro_config = MacroRiskConfig.from_mapping(
        raw.get("macro_event_risk", {}),
        default_cache_path=default_macro_cache,
    )
    macro_risk = MacroRiskController(macro_config) if macro_config.enabled else None
    notifier = None
    if reporter is not None:
        try:
            email_config = EmailNotificationConfig.from_mapping(
                raw.get("email_notifications", {}),
                default_timezone=str(raw.get("report_timezone", "Asia/Shanghai")),
                default_state_path=str(selected_report_dir / "email_notification_state.json"),
            )
            notifier = EmailNotifier(email_config)
        except Exception as error:  # Invalid optional email config cannot prevent trading startup.
            logging.getLogger(__name__).exception("email notifications disabled because configuration is invalid")
            notifier = EmailNotifier(
                EmailNotificationConfig(enabled=False),
                initial_error=f"邮件配置无效，通知已隔离禁用：{error}",
            )
    return TradingEngine(
        adapter,
        strategy,
        risk,
        engine_config,
        reporter=reporter,
        macro_risk=macro_risk,
        notifier=notifier,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-exchange BTC perpetual futures bot")
    parser.add_argument("--config", default="config.example.json")
    parser.add_argument("--exchange", choices=["okx", "binance", "gate"], help="run only one exchange")
    parser.add_argument("--once", action="store_true", help="run one market-data cycle")
    parser.add_argument("--allow-live", action="store_true", help="required extra acknowledgement for mode=live")
    parser.add_argument(
        "--allow-production-live",
        action="store_true",
        help="second acknowledgement required for Binance production live mode",
    )
    parser.add_argument("--check-private", action="store_true", help="read private account equity; never places an order")
    parser.add_argument("--web", action="store_true", help="start the local dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_dotenv()
    if args.web:
        from .dashboard import run_dashboard

        run_dashboard(args.config, host=args.host, port=args.port)
        return
    config = load_config(args.config)
    if config.get("mode", "paper") == "live" and not args.allow_live:
        raise SystemExit("Live mode is blocked unless --allow-live is explicitly provided.")
    names = [args.exchange] if args.exchange else [name for name, item in config.get("exchanges", {}).items() if item.get("enabled")]
    if not names:
        raise SystemExit("No enabled exchange. Edit the config file first.")
    production_live = any(
        name == "binance"
        and config.get("exchanges", {}).get(name, {}).get("environment") == "production"
        for name in names
    ) and config.get("mode", "paper") == "live"
    if production_live and not args.allow_production_live:
        raise SystemExit(
            "Binance production live mode is blocked unless --allow-live and "
            "--allow-production-live are both provided."
        )
    selected_report_dir = (
        report_directory(config, names[0])
        if len(names) == 1
        else Path(config.get("report_dir", "reports"))
    )
    reporter = TradeReporter(selected_report_dir, config.get("report_timezone", "Asia/Shanghai"))
    engines = [build_engine(name, config, reporter=reporter) for name in names]
    if args.check_private:
        for engine in engines:
            print({"exchange": engine.adapter.name, "equity": engine.adapter.fetch_equity()})
        return
    if config.get("mode", "paper") == "live":
        for engine in engines:
            engine.prepare_live()
    if args.once:
        for engine in engines:
            print(engine.evaluate_once())
        return
    if len(engines) == 1:
        engines[0].run_forever()
        return
    # Keep exchange loops independent so one venue outage does not stop the others.
    import threading

    threads = [threading.Thread(target=engine.run_forever, name=engine.adapter.name, daemon=True) for engine in engines]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


if __name__ == "__main__":
    main()
