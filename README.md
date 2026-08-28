# BTC 合约自动化交易系统（第一版）

这是一个“干跑优先”的 BTC 永续合约交易系统骨架，包含：

- 欧易、币安 USDⓈ-M、Gate Futures 三个适配器；
- 1m、5m、1h、4h 多周期趋势/动量信号；
- EMA、RSI、MACD、ATR、突破过滤；
- 按账户风险反推仓位，而不是固定下单量；
- 默认 5% 价格止损、1.5R 止盈、最大名义仓位、单日亏损熔断、连亏冷却；
- `paper`/`dry_run` 默认模式，不会发出真实交易指令。

## 运行

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
Copy-Item .env.example .env
py -m btc_futures_bot.main --config config.example.json --once
```

## 操作日志

本地控制台新增“操作日志”页，记录配置保存、策略/代码变更、引擎启动、停止、重启、首次周期执行和周期错误。日志保存在 `logs/operation_log.jsonl`，支持按日期、类型和关键词筛选；API Key、API Secret、Passphrase、Token 等敏感字段会被过滤，不会写入日志。

策略自动评估任务只在 OKX Demo/paper 模式下工作。未达到样本量和运行时间门槛时只记录评估，不自动修改；满足样本外验证、成本扣除、回撤和无持仓等条件后，才会备份并记录单参数变更，再安全重启引擎。

币安测试网只读检查：

```powershell
python -m btc_futures_bot.main --config config.binance.testnet.json --exchange binance --once
python -m btc_futures_bot.main --config config.binance.testnet.json --exchange binance --check-private
```

第二条命令只读取账户权益，不会下单。确认私有接口正常后，再进行测试网下单开发；实盘模式需要额外的 `--allow-live`，并且不建议在订单成交回报对账完成前开启。

回测 CSV 需要包含 `timestamp,open,high,low,close,volume` 六列，并放在同一目录下命名为 `1m.csv`、`5m.csv`、`1h.csv`、`4h.csv`：

```powershell
py -m btc_futures_bot.backtest .\data\btc
# 同时生成回测交易明细与日/月汇总
py -m btc_futures_bot.backtest .\data\btc --report-dir .\reports\backtest
```

回测统一以 1 分钟数据推进持仓管理；信号只能在对应 K 线收盘后确认，因此入场与反转按下一根 1 分钟 K 线开盘价建模。若开盘已经跳过止损价，止损按更差的开盘价成交，避免用信号当根收盘价或理想止损价高估结果。候选评估还要求完整样本和最新留出段达到最小交易数，并用 bootstrap 置信区间检查单笔期望。

## 交易报表

每次纸面交易平仓后，系统会把记录写入 `report_dir`，默认是 `reports/`：

- `trade_report.csv`：逐笔明细，包括买入时间/价格、卖出时间/价格、方向、数量、毛利、开仓手续费、平仓手续费、总手续费、滑点、资金费、总成本、净利润、净利润率、手续费占成交额比例、手续费占毛利比例、平仓原因和信号分数；
- `daily_summary.csv`：按 `report_timezone` 日期统计交易数、胜率、毛利、手续费、资金费、总成本、净利润、净利润率、平均单笔、最大盈利和最大亏损；
- `monthly_summary.csv`：同样字段按 `report_timezone` 月份统计；默认是 `Asia/Shanghai`；
- `trades.sqlite3`：原始台账，作为 CSV 导出的持久化数据源，适合后续做网页看板或 Excel 查询。

CSV 使用 UTF-8 BOM，Excel 可以直接打开。当前纸面/回测会完整记录已知的平仓成交；实盘中的止损单需要等交易所订单回报确认成交后再写入，不能用“委托价”冒充实际成交价。

报表口径：`net_pnl_pct = 净利润 / 开仓名义金额`，`fee_ratio_pct = 总交易手续费 / 开仓名义金额`，`fee_to_gross_pct = 总交易手续费 / |毛利|`。

将 `config.example.json` 中三个交易所的 `enabled` 改为 `true` 后，程序会分别采集行情并输出信号。为了避免同一策略在三个账户重复暴露，建议第一阶段只启用一个平台做测试；需要切换平台时只改配置。

## 交易策略

当前 `traditional_kline` 策略是多周期趋势跟随：

1. 1h 判断主趋势；
2. 5m 要求金叉/死叉、回踩收复或合格突破，并同时检查 MACD、RSI 和成交量；
3. 1m 提供执行方向确认；
4. 传统信号六项检查必须全部通过才生成信号。

初始止损先按 ATR 计算；当 `structure_stop_lookback_bars` 大于 0 时，再参考最近已收盘触发周期 K 线的结构低点/高点，取两者中更安全的距离，并始终受 `max_stop_loss_pct` 限制。正式配置使用 0.45% 最小距离、最近 1 根 5m K 线且不额外增加 ATR 缓冲，避免把止损放进信号 K 线的正常回踩区间，同时不把风险扩张到更早、更远的摆动低点。

另有默认关闭的 `traditional_allow_1m_impulse` 实验分支，用于弥补强势 5m K 线收盘后已经过度扩张的问题。该分支只使用已收盘的 1m K 线，要求 1h 强趋势、5m MACD/RSI 同向、首次 1m 区间突破、放量、实体/收盘位置合格，并继续使用相对 5m EMA/ATR 的扩张上限；信号按半仓处理。建议至少使用两根 1m K 线确认，并在扣除成本的走步样本外测试通过前保持关闭。

`traditional_failed_breakout_short_shadow` 用于观察“高周期仍偏多，但 5m 出现放量长上影见顶并由下一根放量实体确认下破”的快速反转候选。影子模式只把 `shadow_candidate=short` 和判定原因写入信号日志，不会下单；`traditional_failed_breakout_short_enabled` 才会把它转成真实空头信号。真实分支固定使用半风险，并通过 `traditional_failed_breakout_short_stop_lookback_bars` 和 `traditional_failed_breakout_short_max_stop_loss_pct` 使用更宽的摆动高点保护。该分支属于实验功能，默认保持真实交易关闭。

`traditional_predictive_reversal_short_enabled` 用于处理 5m K 线尚未收盘时已经出现的冲高失败。它只读取已收盘的 1m K 线：先识别靠近 10m/15m 均线簇的放量、超买和 ATR 扩张尖峰，再要求随后首根 1m 同时跌回 EMA9/EMA21 下方、MACD 柱转负、RSI 明显回落且卖出确认量不低于近期均量的配置比例。信号仍然非重绘，不会使用正在形成的 K 线；默认按半风险开空，止损锚定最近 8 根 1m 的冲高结构并受最大止损比例限制。若 1m 时间戳不连续，分支直接保持空仓。

传统 K 线模式还支持三层防追价保护：`traditional_cross_max_extension_atr` 限制 5m 金叉/死叉确认时相对快 EMA 的扩张；`traditional_execution_rsi_*` 与 `traditional_execution_max_extension_atr` 拒绝已经过热/过冷的 1m 执行；`traditional_pressure_filter_enabled` 从已收盘 5m K 线本地合成 10m、15m K 线，并要求入场方向到 EMA/SMA 压力或支撑至少保留 `traditional_pressure_min_room_r` 倍的最小止损空间。合成过程不会增加交易所行情请求。

正式环境的签名持仓对账由 `live_reconciliation_seconds` 节流，仪表盘私有快照由 `dashboard_snapshot_seconds` 缓存。默认均为 5 秒；`candle_refresh_seconds` 可分别限制 1m、5m、1h 公共 K 线请求频率，正式配置默认为 1/3/15 秒，降低短时间重复请求引发 HTTP 429 的概率。行情评估仍可按更短的 `poll_seconds` 运行，交易所硬止损不依赖轮询。

风险配置支持 `loss_streak_pause_minutes`。当 `max_consecutive_losses` 大于 0 且连续亏损达到该值时，引擎进入有期限的长暂停；正式交易重启后会从持久化交易报表恢复最近连亏次数与剩余暂停时间。将 `max_consecutive_losses` 设为 `0` 会完全禁用连亏入场阈值，但单笔亏损后的 `cooldown_minutes` 短冷却仍然生效。

这是一套可验证的基线，不代表收益保证。实盘前必须做历史回测、走样本外测试、测试网运行，并检查手续费、资金费率、滑点、最小下单量、合约面值和强平价格。

## 买入和卖出如何判断

程序不是根据“某一个指标显示涨”就买入，而是对多空分别打分：

- 做多：4h 收盘价在 EMA200 上方且 EMA50 在 EMA200 上方；1h 和 5m 同时满足 EMA20 > EMA50、MACD 柱为正、RSI 在 50～75；最后 1m 向上突破近 20 根高点；
- 做空：上述条件全部反向，RSI 区间为 25～50，1m 跌破近 20 根低点；
- 每个周期最多贡献 2 分，达到 `min_score=6` 且多空分差明确才开仓；否则保持空仓；
- 多周期趋势反向时，先平当前仓位，再等待新方向的确认；止损触发平仓，纸面模式还会按 1.5R 止盈。

因此它判断的是“趋势、动量和突破同时确认”，不是保证涨跌的预测器。

## 手续费、滑点和资金费率

`costs` 使用小数表示费率：`0.0005` 等于 0.05%。默认按 Taker 估算，因为市价开仓和止损市价平仓通常会吃单；如果实际使用挂单成交，需要改成 Maker 并单独验证成交率。

每笔交易的成本估算为：

`开仓手续费 + 平仓手续费 + 开仓滑点 + 平仓滑点 + 预计资金费`

系统会在三处考虑它：

1. 按“止损价可能亏损 + 双边手续费/滑点/资金费”计算仓位，使单笔最大风险仍接近 `risk_per_trade`；
2. 开仓前计算止盈后的净收益，净收益不足 `min_net_edge_pct` 时直接跳过信号；
3. 回测和纸面交易按净 PnL 扣除成本，而不是只看裸价格差。

费率必须改成你账户的实际费率。OKX 可通过账户 `trade-fee` 接口查询，Binance 有 User Commission Rate 接口，Gate 有 Futures Fee 接口；平台官方说明也明确区分 Maker/Taker，并且永续合约还会产生资金费。[OKX 费率接口](https://www.okx.com/docs-v5/en/)、[Binance User Commission Rate](https://developers.binance.com/docs/derivatives/usds-margined-futures/account/rest-api/User-Commission-Rate)、[Gate Futures Fee](https://www.gate.com/docs/developers/apiv4/en/futures/)、[OKX 合约费用说明](https://www.okx.com/en-us/help/how-to-calculate-the-contract-transaction-fee)。

## 安全边界

- API Key 只开读取和合约交易权限，关闭提现；
- 绑定固定 IP；
- 不把 `.env` 提交到 Git；
- `stop_loss_pct=0.05` 是价格波动止损，不是账户亏损 5%。在杠杆合约中，5% 价格逆向波动可能对应更大的保证金损失，甚至先于止损发生强平；
- 欧易和 Gate 的 `contract_size` 必须按平台合约详情接口返回值填写；示例中的 `0` 会阻止误用错误的合约面值。币安 USDⓈ-M 的数量按 BTC 数量传递；
- `max_leverage` 目前用于仓位上限计算，不会自动修改交易所账户杠杆；在测试网启动前请手动确认三家平台都是逐仓、单向持仓和不高于该值的杠杆；
- 第一版使用 REST 轮询以便容易审计。真正 24/7 实盘还需要把行情和订单回报切到 WebSocket，并加入断线重连、订单状态对账和进程守护。

## 交易所测试环境

- OKX：Demo Trading，私有请求需要 `x-simulated-trading: 1`；
- Binance：USDⓈ-M Futures Testnet；
- Gate：独立 Futures TestNet API Key 与 `https://api-testnet.gateapi.io/api/v4`。

适配器中的真实下单函数已隔离，但主循环仍只在显式设置 `mode=live` 且凭据完整时调用；没有提供凭据时会直接拒绝私有请求。

## 本地控制台

启动本地页面：

```powershell
$env:PYTHONPATH="src"
& "C:\Users\bryan_hugo\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -m btc_futures_bot.main --config config.binance.testnet.json --web
```

打开 `http://127.0.0.1:8787`。页面可以保存测试网 API Key、启动/停止引擎、查看实时行情/账户/持仓/挂单，以及按日期筛选逐笔和日/月报表。凭据保存到被 Git 忽略的 `config.binance.testnet.local.json`；默认模式仍是 `paper`。真实测试网下单前仍需完成订单成交回报对账，不要直接把 `live` 当作实盘部署。
