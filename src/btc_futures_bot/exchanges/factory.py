from __future__ import annotations

from typing import Any

from .base import ExchangeAdapter, ExchangeSettings
from .binance import BinanceAdapter
from .gate import GateAdapter
from .okx import OkxAdapter


def make_adapter(name: str, raw: dict[str, Any], account: dict[str, Any]) -> ExchangeAdapter:
    settings = ExchangeSettings(
        name=name,
        environment=str(raw.get("environment", "testnet")),
        base_url=str(raw["base_url"]),
        symbol=str(raw["symbol"]),
        api_key_env=str(raw.get("api_key_env", "")),
        api_secret_env=str(raw.get("api_secret_env", "")),
        passphrase_env=str(raw.get("passphrase_env", "")),
        settle=str(raw.get("settle", "usdt")),
        margin_mode=str(account.get("margin_mode", "isolated")),
        position_mode=str(account.get("position_mode", "net")),
        contract_size=float(raw.get("contract_size", account.get("contract_size", 1.0))),
    )
    adapters = {"okx": OkxAdapter, "binance": BinanceAdapter, "gate": GateAdapter}
    try:
        return adapters[name](settings)
    except KeyError as error:
        raise ValueError(f"unsupported exchange: {name}") from error
