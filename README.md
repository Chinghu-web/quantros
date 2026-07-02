# QuantROS 策略实盘就绪度证伪器

回答一个问题:**这条漂亮的回测曲线,到底能不能实盘?** 不审代码,只审行为。

**定位铁律:本平台是【证伪器】,不是保证书。** 它能高置信地说"不能实盘"
(被某道门证伪);最强正向结论只是"未被任何已知测试证伪",永不承诺盈利。

## 产品架构:通用层当门 + 时点化沙箱当核 + 五门判决

```
任何策略(期权/多因子/多腿/任意框架)
   │  只交"各组合日净收益"(纯数字,无代码)
   ▼
通用层正门 quantros-gate ── ②PBO ③DSR ──→ ⚠️ 初筛判决(未验证前视,非实盘结论)
   │  想要可信判决?策略进沙箱重放
   ▼
时点化沙箱 sandbox ── ①诚实:ctx.history() 只喂 ≤t 数据,前视物理不可能
   │  产出【可信】收益曲线 → ②③在其上才有意义
   ▼
五门判决 final_verdict:
  edge 门  ①沙箱(诚实) ②PBO(非过拟合) ③DSR(净edge显著,校正搜索次数)
  风险门  ④尾部/压力 ⑤容量
  ①②③过 = "真实、诚实、非过拟合的净 edge";五门全过 = "未被证伪,可考虑实盘"
```

三道 edge 门缺一不可(各有断言为证):②是相对排名,会放过"稳定亏钱的家族",
③补绝对门;①缺失时②③在撒谎的曲线上照样通过(垃圾进垃圾出),判决强制降级"初筛"。
沙箱自身经过物理自证:扰动未来行情,过去的每一个决策逐位不变(回归测试钉死)。

此外还有对向量化策略的**五柱深度体检**(因果/过拟合/成本/容量/稳健),逐维给
✅通过 / ❌硬伤 / ⚪未评估,刻意不给会误导人的单一总分。

| 柱子 | 问的问题 | 方法 |
|---|---|---|
| 一 · 因果性 | 偷看未来没有? | 对未来注入扰动,看过去信号是否改变(物理可证) |
| 二 · 过拟合 | 真 edge 还是调参运气? | 样本外退化 + PBO(CSCV) |
| 三 · 成本 | 扣手续费/滑点还剩多少? | 盈亏平衡成本 + 换手率 |
| 四 · 容量 | 资金做大会不会自杀? | 平方根冲击律(模型估计,非物理真值) |
| 五 · 稳健性 | 只在那条历史上活,还是换个世界也活? | E1 block bootstrap 重采样 + E2 压力注入 |

> 核心理念:每根柱子都标注**它对什么失明**;"未评估 ≠ 通过"。
> 一个诚实的"我判断不了"远比一个虚假的"85 分"安全。

## 安装与运行
```bash
pip install -e .              # 安装(可 pip install -e ".[data]" 带 akshare 真实数据)
quantros                      # ★ 统一入口:漏斗指南(你有什么 → 走哪条路)
pytest -q                     # 应全绿
python -m quantros.demo       # 看现场拦截演示
```

聚宽用户一条命令(取数缓存 → 零改写沙箱 → ②③ → 五门判决 → HTML 报告):
```bash
JQ_USER=手机号 JQ_PASS=密码 quantros jq 我的策略.py \
    --data 510300.XSHG --start 2023-01-01 --end 2025-12-31 \
    --grid "fast=5,10;slow=20,40,60"        # 必须是全部搜索过的参数(诚实义务)
```
其余路径:`quantros gate returns.csv`(收益初筛)/ `quantros positions ...`(持仓)/ `quantros check`(自检)。
契约写错时沙箱给**带示例的中文报错**(如 on_bar 返回类型、weight 非数字)。

## 用法(本地,代码不出门)
完整模式——把你的向量化策略类交给 `diagnose`,全程本地 import:
```python
import quantros
from quantros.data import load_csv

data = load_csv("index_futures.csv")        # trading_date,symbol,close[,volume/adv]
quantros.diagnose(MyStrategy, data, profile="cn_index_futures")  # 五柱满血(按品种标定阈值)
```
中档模式——别的框架,只能导出持仓时:
```bash
python -m quantros.multiconfig positions.csv prices.csv --out report.json
```
可选:把判决(仅指标)同步到云端,代码/数据永不离开本地:
```python
from quantros.sync import HTTPSink
quantros.diagnose_outputs(positions, prices, report_sink=HTTPSink(url, api_key))
```
真实数据接入(可选依赖):`pip install akshare` 后 `quantros.data.from_akshare()`。

## 架构
**柱子一 · 因果性（策略有没有偷看未来）**
- `quantros/causality.py` — 双探针测谎仪核心（尖峰探针 + 重采样探针 + 探针隔离开关）
- `quantros/zoo.py` — 作弊策略动物园（每只野兽标注应由哪个探针拦截）
- `quantros/generalization.py` — 探针C：跨市场泛化测试（启发式，针对硬编码作弊）
- `tests/test_causality.py` — 回归套件，含"探针隔离断言"防止巧合绿

**柱子二 · 过拟合（漂亮曲线是真 edge 还是调参运气）**
- `quantros/overfitting.py` — 探针O1 样本外退化(Walk-forward) + 探针O2 回测过拟合概率 PBO(CSCV, Bailey & López de Prado)；含真实-edge 数据生成器与过拟合动物园
- `tests/test_overfitting.py` — 回归套件，含探针分工断言 + PBO 对单参数策略失明的诚实盲区断言

**柱子三 · 成本（扣掉手续费+滑点还剩多少 edge）**
- `quantros/cost.py` — 探针C1 盈亏平衡成本(Breakeven Cost) + 换手率 + 净夏普 vs 成本扫描；含成本动物园
- `tests/test_cost.py` — 回归套件，含低换手 vs 高频空转的分离断言 + 不交易策略失明的退化边界断言

**柱子四 · 容量（资金做大会不会自杀）**
- `quantros/capacity.py` — 探针D1 容量(平方根冲击律 Almgren-Chriss) + 净夏普 vs 规模扫描；含流动性动物园
- `tests/test_capacity.py` — 回归套件，含容量随流动性同向缩放的分离断言 + 不交易策略失明的边界断言
- ⚠️ 注意：容量是【模型估计】，依赖冲击系数 k 与 √ 律假设，给数量级参考，非物理真值

**柱子五 · 状态稳健性（只在那条历史上活,还是换个世界也活）**
- `quantros/robustness.py` — 探针E1 同步 block bootstrap 重采样(看净夏普分布最差分位) + 探针E2 压力注入(放大波动 / 持续崩盘);含稳健性动物园
- `tests/test_robustness.py` — 回归套件，含 E1/E2 分工断言(满仓多 E1 放行/E2 拦截) + block 盲区断言(block=1 退化 IID 会误杀真策略)

**压轴 · 体检报告（五柱合一的产品出口）**
- `quantros/report.py` — `health_report(strategy_cls, df)` 跑五柱,逐维给 ✅/❌/⚪,总评只回答"有无硬伤 + 哪些维度没评到",**刻意不给单一总分**
- `tests/test_report.py` — 验证判决分流正确 + "BLIND 绝不被当成 PASS"的诚实总评断言

**中档 · outputs-only 诊断（无需源码的获客钩子）**
- `quantros/multiconfig.py` — `diagnose_multiconfig(positions_by_config, prices)`:只需【多配置持仓+价格】,不交源码即可做 PBO(关键:PBO 只需收益矩阵)+ 满血成本/容量
- CLI: `python -m quantros.multiconfig positions.csv prices.csv --out report.json`(本地跑,不联网)
- 连接接缝: `report_sink` 回调只收到【判决+指标】,代码/原始持仓永不离开本地——"本地跑也能连云端、却不泄露 IP"
- `tests/test_multiconfig.py` — "从持仓恢复 PBO"断言 + "接缝只送判决不送 IP"断言
- ⚠️ 本档够不到 一·因果 / 五·稳健(需完整模式重跑策略);定位为"快速冒烟测试"

### 输入分层(本地 pip,完整模式不需上传)
| 档 | 用户给什么 | 解锁 |
|---|---|---|
| 主路 | `generate_signals(df)` 契约 + 数据(本地) | 五柱满血 |
| **中档** | 多配置持仓序列 + 价格 | 成本+容量+**真 PBO**(无需源码) |
| 兜底 | 单组持仓+价格 | 成本+容量+因果"闻味" |

**正门 · 通用层(任何策略类型,收益序列即可)**
- `quantros/universal.py` — ②PBO + ③Deflated Sharpe(按申报搜索次数 n_trials 校正选择效应,无 scipy 依赖)+ block bootstrap;无沙箱背书时判决强制"初筛"
- CLI: `quantros-gate returns.csv --n-trials 500`(长表 combo,date,ret;须含全部试过的组合、净收益)
- `tests/test_universal.py` — 含"稳定亏钱家族骗过②被③拦"的结构性盲区断言、"申报次数越多 DSR 越低"断言、"无沙箱判决必降级"断言

**核 · 时点化沙箱(①诚实门,前视物理不可能)**
- `quantros/sandbox.py` — 事件式逐日重放:`on_bar(ctx)` 只能经 `ctx.history()` 拿 ≤t 数据,未来行不在上下文里;`run_sandbox_grid` 直通 ②③;记账复用成本柱口径
- `tests/test_sandbox.py` — **沙箱自证**:扰动后 30% 行情,前 70% 决策逐位不变(可复现实验);贪婪索取 history 只得历史前缀;参差面板支持
- ⚠️ 边界:保证【时点诚实】不保证【数据诚实】(幸存者偏差/错价需数据审计层);MVP 为日频收盘价,期权链/分钟频属家族扩展

**出口 · 五门判决(证伪器)**
- `quantros/verdict.py` — edge 门①②③ + 风险门④⑤,分级:已证伪 / 初筛 / 仅有 edge(不构成实盘依据)/ 风险被证伪 / 未被证伪;最强结论也带"不保证盈利"
- `tests/test_verdict.py` — 每个分级逐条钉死,含"最强结论必带免责"断言

**家族扩展 · 期权卖方(第一个真实迁移案例:卖出宽跨式)**
- `quantros/options.py` — 时点化期权链上下文(`ctx.chain()/otm_leg()/price()/dte()`,与主沙箱同一物理保证,有扰动自证测试)+ 期权沙箱(到期内在价值结算/每手费用)+ **④期权版:持仓账本 BSM 重定价**(收盘价反解隐波 → 跳空×隐波跳升情景)+ 自带 BSM/隐波(零依赖)
- `ShortStrangle` — 从真格 POBO 版 1:1 移植的决策核(权利金均线择时/双卖虚值/止盈止损/DTE移仓/冷却),差异诚实标注在 docstring
- `tests/test_options.py` — 含头条断言 **"1手 vs 5手夏普完全相同,④一个过一个爆"**——仓位才是尾部风险,夏普看不见它
- ⚠️ 边界:当前用【合成链】(平价 BSM 单一隐波,无微笑/盘口)验证管道与探针逻辑,**不用于对真实市场下结论**;真实时点化链(聚宽导出)接入后同一套代码直接可用;保证金/强平未建模
- `quantros/jq_bridge.py` — 聚宽真实链桥:一条命令 取数(缓存 parquet,额度只花一次)→ 宽跨式过五门;凭证只进用户终端。首跑:`JQ_USER=手机号 JQ_PASS=密码 python3 -m quantros.jq_bridge`,之后离线用缓存;`tests/test_jq_bridge.py` 离线钉死下游全路径

**平台基础设施**
- `quantros/__init__.py` — 顶层入口 `diagnose()`(完整模式) / `diagnose_outputs()`(中档)
- `quantros/data.py` — 数据接入层:规范 schema + CSV 加载 + akshare/tushare 适配器(可选依赖);**`from_jq()` 聚宽通用行情**(股票/ETF/指数日线,凭证只进终端,本地 parquet 缓存额度只花一次,成交额直接作 adv);`tests/test_data.py` `tests/test_from_jq.py`
- `quantros/htmlreport.py` — HTML 体检报告:单文件自包含,逐门着色,免责与"未评估≠通过"固定出现不可删;`quantros-gate --html` / jq_bridge 自动产出;`tests/test_htmlreport.py`
- `quantros/sync.py` — 云端接缝:JSONFileSink / HTTPSink / License(API key + 离线宽限),传输层依赖注入可测;**只送判决不送 IP** 有断言为证;`tests/test_sync.py`
- `quantros/profiles.py` — 品种参数档:`cn_index_futures`(成本 3bp,非股票的 10bp)等;阈值显式化、可被 opts 覆盖;`tests/test_profiles.py`
- `quantros/check.py` — 真实数据端到端核对(`quantros-check`):证明因果柱在真实数据上仍能抓泄漏
- `pyproject.toml` — `pip install -e .`;命令行入口 `quantros-check` / `quantros-multiconfig`

## 🛡️ 诚实的边界清单（Causality Frontier）
本测谎仪**不承诺 100% 防住所有泄漏**。经实跑验证：

### ✅ 已验证可拦截并定位
| 作弊形态 | 拦截探针 |
|---|---|
| 时序位移 `shift(-1)` | 尖峰探针 |
| 全样本 mean/std 标准化 | 尖峰探针 |
| 品种内全样本排名 `rank().over(symbol)` | 重采样探针（尖峰探针对此**失明**，有隔离断言为证）|
| 全量 fit 的 ML 模型 | 双探针（全局依赖链，对任何扰动敏感）|

### ⚠️ 已知盲区（动态扰动的理论天花板）
| 盲区 | 现状 | 应对 |
|---|---|---|
| **硬编码未来常数** | 双探针**必然失明**（有断言记录在案）| 探针C 跨市场泛化测试可启发式标记（非物理证明，有假阳性）|
| 未来 universe 成分泄漏 | 部分可被重采样探针捕捉 | 待扩充 |
| 跨时间戳非同步对齐 | 未覆盖 | 待扩充 |

## 🛡️ 过拟合柱子的诚实边界（Overfitting Frontier）
### ✅ 已验证可区分
| 形态 | 拦截探针 |
|---|---|
| 样本内调参、样本外坍塌 | O1 样本外退化 |
| 参数族里挑出的"运气冠军" | O2 PBO（CSCV，给出过拟合概率 ≈0.5） |
| 有真 edge 的稳健策略 | 双探针一致**放行**（参数高原 / OOS 稳定 / PBO≈0） |

### ⚠️ 已知盲区
| 盲区 | 现状 |
|---|---|
| **无 param_grid 的单一手调策略** | O2 PBO **必然失明**（无可观测搜索空间，有断言记录在案）|
| 区分"过拟合" vs "市场状态切换" | O1 单切分无法分辨；需更长样本 / 多段 walk-forward |
| 真实手续费 / 滑点对 edge 的侵蚀 | 已覆盖（见下方柱子三）|

## 🛡️ 成本柱子的诚实边界（Cost Frontier）
### ✅ 已验证可区分
| 形态 | 拦截探针 |
|---|---|
| 毛收益正、换手过高、净收益被成本吃光 | C1 盈亏平衡成本（毛夏普≈2 仍被打成净负） |
| 低换手、edge 真、扛得住真实成本 | C1 **放行**（盈亏平衡成本远高于真实成本） |

### ⚠️ 已知盲区
| 盲区 | 现状 |
|---|---|
| **从不交易的策略** | 无换手 → 盈亏平衡=∞,探针对其失明（不付成本也不赚钱,有断言记录在案）|
| 真实成本随品种 / 时段变化 | `realistic_bps` 需按标的设定,非自动 |
| **冲击成本 / 资金容量** | 已覆盖（见下方柱子四）|

## 🛡️ 容量柱子的诚实边界（Capacity Frontier）
### ✅ 已验证可区分
| 形态 | 拦截探针 |
|---|---|
| 同一信号跑在低流动性品种,小规模即被冲击吃光 | D1 容量（容量随 ADV 同向缩放） |
| 跑在高流动性品种,容量远超目标资金 | D1 **放行** |

### ⚠️ 已知盲区（与前三根柱子不同：容量是模型估计，非物理真值）
| 盲区 | 现状 |
|---|---|
| **容量绝对值依赖冲击系数 k 与 √ 律假设** | 给数量级参考；k 需按市场校准 |
| **从不交易的策略** | 容量=∞,探针对其失明（有断言记录在案）|
| **有交易但无毛 edge** | 容量 N/A(三态:PASS / FAIL / ⚪N/A),绝不当成"通过"(有断言记录在案)|
| 日内拆单 / 多日建仓降低冲击 | 未建模（保守估计,倾向低估容量）|

> **置信度提示(针对真实弱 edge)**:过拟合柱报样本外夏普的 **t 值**(|t|<2 提示"样本不足以判定")与 **PBO 配置数**(N<10 提示"PBO 偏粗");稳健柱报"为正占比"的**二项标准误**(贴近阈值则提示"置信不足")。贴着噪声的判决会自报家门,不假装确定。

## 🛡️ 稳健性柱子的诚实边界（Robustness Frontier）
> 根本边界:**合成数据只能检验你放进生成器的那种变化。** 生成器的假设,就是这根柱子的边界。

### ✅ 已验证可区分
| 形态 | 拦截探针 |
|---|---|
| 无真 edge,换条平行历史就坍塌 | E1 重采样（最差分位净夏普 ≤0） |
| 平时稳健、却死于没见过的崩盘 regime | E2 压力（满仓多 E1 放行 / E2 拦截——分工断言为证） |
| 真 edge + 低换手 | 双过（重采样下稳定为正,压力情景下不爆） |

### ⚠️ 已知盲区
| 盲区 | 现状 |
|---|---|
| **E1 造不出数据里没有的 regime** | 牛市历史重采样永远采不出崩盘——这正是 E2 存在的理由 |
| **block 块长是关键旋钮** | 太小(→IID)毁掉自相关误杀真策略、太大只复读原路径(有断言记录在案) |
| **E2 结论只在所选压力情景下成立** | 情景(波动×2 / 持续崩盘)由人设定,非数据自带 |
| 合成数据可能给虚假安全感 | 判据用"最差分位存活",不用"平均为正" |

**探针 E3 · 历史状态切片(`quantros/regime.py`)——"各市况都稳"的可证伪版本**
- 市场状态 = 滚动波动(高/低)×趋势(上/下)逐日打标;任一已见状态内累计亏损超阈值 → 证伪("设计押错市况")
- 实证:马丁格尔摊平在 4 个状态中 3 个为正、整体曲线好看,'下行/高波'一刀 -80% → 拦截
- `tail_gate(strategy, df)` = E2(假设的未见情景)+ E3(已见历史)→ 直接喂 ④
- ⚠️ **归纳边界(火鸡问题,有分工断言为证)**:E3 只能检验历史里出现过的状态——
  "过去所有状态稳住 ≠ 未知状态能活";平静史上的满仓多 E3 必然放行、由 E2 拦截。
  尤其警惕卖权利金类:样本里没有 vol spike 时 E3 必然全绿,全绿 ≠ 尾部安全。
- 迁移指南:`MIGRATION.md`(聚宽日频策略 30 分钟迁入沙箱,含逐行对照与诚实义务清单)
- **聚宽零改写垫片 `quantros/jqcompat.py`**:实现聚宽同名 API 接沙箱底层,用户策略文件一行不改直接过五门;PIT 保证穿透垫片(自证测试);不支持的 API 大声报错点名,绝不静默假装;调度三件套 run_daily/run_weekly/run_monthly(every_bar=日频每日一次);活账户(cash/持仓随下单变化,堵杠杆叠加);`tests/test_jqcompat.py`

**家族扩展 · 基本面(`quantros/fundamentals.py`)——解锁选股类策略**
- **时点成分股**(月度快照,≤当日最近一期)+ **公告日对齐财报**(聚宽 `get_fundamentals(date=)` 原生按公告日可见)→ 物理堵死幸存者偏差与财报前视
- **聚宽查询 DSL 仿真**(`query/valuation/balance`,含 between/in_/比值过滤/order_by)→ 格雷厄姆选股、小市值轮动、低PB银行等模板**零改写**运行
- **反作弊**:策略查询未来日期的成分/财报 → "前视企图,物理拒绝";面板外字段点名拒绝
- **自证(基本面版)**:打乱未来快照的市值/PB,扰动前持仓决策逐位不变(断言钉死)
- 真实取数:`fetch_fundamental_store(index, start, end)`(凭证进终端,月度快照≈2次调用/月,parquet 缓存后离线);合成市场 `generate_fundamental_market` 供机制验证
- 用法:`run_jq_strategy(src, prices, funda=store)` / `run_jq_grid(..., funda=store)`
- ⚠️ 边界:快照月度(月内变化最长滞后一月,方向保守);聚宽公告日口径由数据商保证,平台不重复审计;`tests/test_fundamentals.py`

> 核心纪律：每加一只野兽，必须实跑验证它被【正确的探针】拦截；
> 每个"通过"都要追问是因正确原因通过、还是巧合通过。
> 永不整篇重写本目录文件——只允许"改一行→pytest→绿了才提交"。
