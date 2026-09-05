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

## 交易模型 2.0（LightGBM Meta Gate）

2.0 保留现有 `traditional_kline` 作为主策略，只在准备新开仓时用 LightGBM 对候选信号做放行或拒绝。仓位计算、硬止损、动态退出、报表、邮件和控制台仍使用原来的独立模块；模型不可用时，`enforce` 模式会禁止新开仓，但不会阻止已有仓位退出。

安装可选模型依赖并下载 Binance USDⓈ-M 公共历史数据：

```powershell
pip install -e ".[model2]"
python .\scripts\download_binance_klines.py --symbol BTCUSDT --start 2025-09-01T00:00:00Z --output-dir .\data\binance_meta_12m
```

按当前有效策略、风险和费用配置重放候选，训练模型并冻结验证集阈值：

```powershell
python .\scripts\train_meta_model.py --config .\config.binance.model2.json --data-dir .\data\binance_meta_12m --output-dir .\artifacts\trade_model_2_0
```

训练和实时推理共用同一特征实现。制品会校验特征顺序、模型文件、完整策略配置以及风险/费用/标签/历史窗口政策；任一指纹不一致都会拒绝加载。当前固定 TP/SL 三重障碍标签与实盘动态退出并不完全等价，因此生成的模型仅供 `paper` 对比，不会自动获得 `approved_for_live`。

仓库当前附带的 `meta-20250901-20260904-v1` 是管线验证制品，不是已证明盈利的模型：其冻结阈值在 holdout 只选择 10/650 个候选，扣成本期望为 -0.003054、Profit Factor 为 0.266，因此明确标记为 `statistically_qualified=false` 和 `approved_for_live=false`。8788 可以用它验证门控、审计和 A/B 数据采集，但不能据此切换真实账户。

在独立端口启动 2.0 纸面交易：

```powershell
.\scripts\start_model2.ps1
```

打开 `http://127.0.0.1:8788`。2.0 使用独立的报表、操作日志、模型决策 SQLite 和邮件状态，不会复用 8787 的运行文件。`trade_model.mode` 支持 `off`、`shadow` 和 `enforce`；正式比较使用 `enforce`，阈值来自训练制品而不是手填概率。

若以后改为两个真实账户 A/B 并跑，必须使用不同 Binance 账户或子账户、不同 API Key 和 `BINANCE_MODEL2_API_KEY` / `BINANCE_MODEL2_API_SECRET` 环境变量。同一个单向持仓账户不能同时运行两套机器人，因为它们会共同看到并操作同一净仓位和保护单。比较时优先看净 R、Profit Factor、扣费期望、最大回撤、候选覆盖率和执行错误率，而不是只看胜率。

已有成交记录可先做实盘尸检：

```powershell
python .\scripts\analyze_trades.py .\reports\binance-production\trade_report.csv
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

初始止损先按 ATR 计算；当 `structure_stop_lookback_bars` 大于 0 时，再参考最近已收盘触发周期 K 线的结构低点/高点，取两者中更安全的距离，并始终受 `max_stop_loss_pct` 限制。

`min_stop_cost_multiple` 再补一层成本下限：止损距离不得小于 `2 × (手续费 + 滑点) × 该倍数`。仓位是按止损距离反推的，所以把止损收窄并不会少付手续费，只会把同样的成本摊到更小的 1R 上——`往返成本 ÷ 止损比例` 就是每笔交易开仓即让给交易所的那部分 R。0.05% taker 加 0.02% 滑点等于 0.14% 往返成本，配 0.25% 止损时成本占 0.56R，需要约 78% 胜率才能在 1:1 出场下打平；倍数取 3.0 把止损抬到 0.42%，成本降到 0.33R。该下限仍受各分支 `max_stop_loss_pct` 上限约束，设为 `0` 即关闭。正式配置使用 0.45% 最小距离、最近 1 根 5m K 线且不额外增加 ATR 缓冲，避免把止损放进信号 K 线的正常回踩区间，同时不把风险扩张到更早、更远的摆动低点。

另有默认关闭的 `traditional_allow_1m_impulse` 实验分支，用于弥补强势 5m K 线收盘后已经过度扩张的问题。该分支只使用已收盘的 1m K 线，要求 1h 强趋势、5m MACD/RSI 同向、首次 1m 区间突破、放量、实体/收盘位置合格，并继续使用相对 5m EMA/ATR 的扩张上限；信号按半仓处理。建议至少使用两根 1m K 线确认，并在扣除成本的走步样本外测试通过前保持关闭。

`traditional_failed_breakout_short_shadow` 用于观察“高周期仍偏多，但 5m 出现放量长上影见顶并由下一根放量实体确认下破”的快速反转候选。影子模式只把 `shadow_candidate=short` 和判定原因写入信号日志，不会下单；`traditional_failed_breakout_short_enabled` 才会把它转成真实空头信号。真实分支固定使用半风险，并通过 `traditional_failed_breakout_short_stop_lookback_bars` 和 `traditional_failed_breakout_short_max_stop_loss_pct` 使用更宽的摆动高点保护。该分支属于实验功能，默认保持真实交易关闭。

`traditional_predictive_reversal_short_enabled` 用于处理 5m K 线尚未收盘时已经出现的冲高失败。它只读取已收盘的 1m K 线：先识别靠近 10m/15m 均线簇的放量、超买和 ATR 扩张尖峰，再要求随后首根 1m 同时跌回 EMA9/EMA21 下方、MACD 柱转负、RSI 明显回落且卖出确认量不低于近期均量的配置比例。信号仍然非重绘，不会使用正在形成的 K 线；默认按半风险开空，止损锚定最近 8 根 1m 的冲高结构并受最大止损比例限制。若 1m 时间戳不连续，分支直接保持空仓。

超短分支可按方向停用不稳定的例外入场：`traditional_ultra_short_reversal_allow_long/short` 控制 1m 极值收复/拒绝反转，`traditional_ultra_short_countertrend_allow_long/short` 控制是否允许逆已收盘 1h 强趋势入场。正式配置关闭多空两个方向的逆 1h 强趋势入场；这不会影响顺势延续，也不会影响通过独立反转结构确认的信号。

超短线默认使用 `traditional_ultra_short_trailing_trigger_r/distance_r` 管理追踪止盈；`traditional_ultra_short_reversal_short_trailing_trigger_r/distance_r` 可为 `1m_ultra_short_reversal_short` 单独设置更早、更紧的保护，不影响趋势延续等其他信号。

`enable_adverse_dynamic_exit` 在亏损阶段启用本地动态软止损。仓位亏损达到 `adverse_dynamic_exit_trigger_r`，并且连续 `adverse_dynamic_exit_confirmation_bars` 根已收盘 1m K 线同时出现价格与 MACD 反向时，程序在线市价退出。软止损不撤销或移动交易所原始硬止损；网络或进程中断时硬止损继续兜底。

传统 K 线模式还支持三层防追价保护：`traditional_cross_max_extension_atr` 限制 5m 金叉/死叉确认时相对快 EMA 的扩张；`traditional_execution_rsi_*` 与 `traditional_execution_max_extension_atr` 拒绝已经过热/过冷的 1m 执行；`traditional_pressure_filter_enabled` 从已收盘 5m K 线本地合成 10m、15m K 线，并要求入场方向到 EMA/SMA 压力或支撑至少保留 `traditional_pressure_min_room_r` 倍的最小止损空间。合成过程不会增加交易所行情请求。

Binance 正式环境的最新成交价、标记价格及 1m/5m/1h K 线使用公共 WebSocket，账户、持仓、普通订单和条件单使用 User Data Stream。私有流在连接或重连时只读取一次完整账户/挂单 REST 快照，持仓直接复用账户快照中的 `positions`，代码不再请求 `/positionRisk`；之后由 `ACCOUNT_UPDATE`、`ORDER_TRADE_UPDATE` 和 `ALGO_UPDATE` 增量维护。`live_reconciliation_seconds` 控制的是本地缓存对账频率，不会触发私有 REST 轮询。listenKey 按 Binance 要求每 30 分钟续期。仪表盘每秒从内存读取最新成交价、标记价、持仓和订单；持仓标记价与未实现盈亏也按最新标记价重算，不受 `dashboard_snapshot_seconds` 的慢快照周期影响，同时不会增加交易所 REST 压力。计划内重启会持久化机器人管理的仓位，只有交易所仓位与唯一的 `btcbot-stop-*` 硬止损在方向、数量、入场价、订单 ID 和触发价上全部吻合时才恢复管理，否则仍拒绝启动。行情评估仍可按更短的 `poll_seconds` 运行，交易所硬止损不依赖本地轮询。

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
- Binance 行情和订单/账户回报均已切到 WebSocket，并带断线重连、listenKey 续期及重连快照对账。REST 仍用于启动/重连快照、交易写操作和异常结果确认；这些操作是交易所协议的一部分，不进行周期性轮询。

## 交易所测试环境

- OKX：Demo Trading，私有请求需要 `x-simulated-trading: 1`；
- Binance：USDⓈ-M Futures Testnet；
- Gate：独立 Futures TestNet API Key 与 `https://api-testnet.gateapi.io/api/v4`。

适配器中的真实下单函数已隔离，但主循环仍只在显式设置 `mode=live` 且凭据完整时调用；没有提供凭据时会直接拒绝私有请求。

## 本地控制台

邮件通知启用且配置完整时，系统会把可捕获的引擎周期异常、实盘启动预检失败、开仓/成交确认/保护单/紧急平仓/主动平仓错误，以及 Binance HTTP 418、429、`-1003` 或进程内限频状态作为紧急邮件优先发送。同一持续事故在 30 分钟内去重，持续未恢复时每 30 分钟最多提醒一次；确认恢复后会立即重新武装，同类故障再次发生会重新告警。IP 被限制或封禁时程序不会自动切换代理节点，邮件会提示人工切换节点并重启引擎。

紧急邮件由交易程序异步发送，不会阻塞订单安全处理。若 Python 进程被强制终止、电脑断电、整机断网或 SMTP 本身不可用，进程内通知无法送达；需要覆盖这些情况时，应另配独立于交易进程的系统级监控。

启动本地页面：

```powershell
$env:PYTHONPATH="src"
& "C:\Users\bryan_hugo\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -m btc_futures_bot.main --config config.binance.testnet.json --web
```

打开 `http://127.0.0.1:8787`。页面可以保存测试网 API Key、启动/停止引擎、查看实时行情/账户/持仓/挂单，以及按日期筛选逐笔和日/月报表。凭据保存到被 Git 忽略的 `config.binance.testnet.local.json`；默认模式仍是 `paper`。

Windows 手动重启可以直接双击 `scripts\restart_bot.cmd`，也可以在 PowerShell 中运行：

```powershell
.\scripts\restart_bot.ps1
```

脚本会先读取页面状态并记录 Binance 实盘快照。空仓时必须没有挂单；持仓时必须恰好有一个方向、数量和价格均匹配的机器人 `STOP_MARKET` 保护单。停止引擎后、启动新引擎前和新引擎完成首个周期后都会重新核对这份快照；连接不健康、状态 stale、出现额外仓位/挂单或保护单发生变化时均会 fail closed。若原页面已经完全宕机，脚本会启动一个尚未运行引擎的新页面，等待其私有状态同步并通过同样的实盘快照校验，再以该快照为 cold-start 基线启动引擎。最终校验失败时脚本会停止引擎并要求人工核对，不会主动提交恢复仓位的订单。

只运行 GET 状态校验、不发送 POST 或停止进程时，使用 `.\scripts\restart_bot.ps1 -CheckOnly`。仅重启页面而不启动交易引擎时，使用 `.\scripts\restart_bot.ps1 -DashboardOnly`。从隔离源码临时运行、但继续使用原实盘目录中的配置、报告和日志时，可传入 `-RuntimeWorkingDirectory C:\my_project`；此时 `PYTHONPATH` 仍指向脚本所在源码树。当前 API 未返回价格 tick 时，脚本只对 Binance `BTCUSDT` 使用明确的 `0.1` 回退；其他品种必须由状态 API 提供 `price_tick`，否则拒绝重启。若切换了网络节点，必须运行重启脚本，旧进程内保存的 Binance 限频倒计时不会因为出口 IP 改变而自动消失。

`*.local.json` 只是覆盖层：加载时先读同名的已提交配置，再把本地文件逐键覆盖上去。页面保存只写它认识的字段，所以本地文件总会落后于仓库配置；如果不做合并，新增的策略开关会静默退回代码默认值（通常是关闭），仓库里明明打开的分支在实盘中并不会执行。合并后本地文件仍然优先，但没写到的键会继承已提交配置。真实测试网下单前仍需完成订单成交回报对账，不要直接把 `live` 当作实盘部署。

### Windows 独立后台运行

启动/重启脚本通过 Windows WMI 服务创建隐藏后台进程，不再把服务挂在 Codex 或终端的进程树下。启动命令结束后可关闭终端或退出/更新 Codex。日志仍在运行目录的 `logs` 中；此方式不提供电脑重启后的自动启动或进程崩溃自动恢复。

请使用独立安装的 Python 3.11+，推荐在各运行目录创建 `.venv` 并执行 `python -m pip install -e .`（模型 2 使用 `python -m pip install -e ".[model2]"`）。脚本拒绝 Codex 路径及基于 Codex Python 创建的虚拟环境，也不会在 WMI 启动失败后回退为终端子进程。配置和凭据应存放在应用支持的本地配置中，后台进程不继承当前终端临时设置的环境变量。

主分支：`powershell -NoProfile -ExecutionPolicy Bypass -File C:\my_project_main_runtime\scripts\restart_bot.ps1 -ConfigPath config.binance.testnet.local.json`。保留原有实盘快照和首轮健康检查。

模型 2：`powershell -NoProfile -ExecutionPolicy Bypass -File C:\my_project\scripts\start_model2.ps1 -Restart`。`-Restart` 会在确认模拟引擎停止后重建后台进程。
