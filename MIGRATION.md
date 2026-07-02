# 聚宽策略 → QuantROS 沙箱:迁移指南

目标:把你的聚宽(JoinQuant)日频策略迁进时点化沙箱,拿到五门判决
(①诚实 ②非过拟合 ③净edge显著 + ④尾部 ⑤容量),全程本地,代码不出门。

## ⚡ 路径 0(首选):零改写——聚宽兼容垫片

**你的策略文件一行不用改。** 垫片实现了聚宽同名 API(`g`/`attribute_history`/
`order_target_value`/`run_daily`/`handle_data`…),底层接时点化沙箱:

```python
from quantros.jqcompat import run_jq_strategy, run_jq_grid
from quantros.universal import evaluate_returns

res = run_jq_strategy("my_jq_strategy.py", data, bps=3.0)        # 直接跑,零改写
rets = run_jq_grid("my_jq_strategy.py", data,                     # 参数族(覆盖到 g)
                   grid=[dict(fast=f, slow=s) for f in (5,10) for s in (20,40,60)])
rep = evaluate_returns(rets, sandbox_verified=True)               # ②③
```

- PIT 物理保证**穿透垫片**(有扰动自证测试:改未来行情,过去决策逐位不变);
- `attribute_history` 忠实保留聚宽"不含当日"语义;
- **不支持的 API 大声报错并点名**(如 get_fundamentals/get_price/分钟频),绝不静默假装——
  报错了再走下面的手工迁移路径;
- grid 是你【真正搜索过的全部参数组合】——少报=③被高估(诚实义务)。

垫片支持子集:日频、close 字段、order_target_value/order_target/order_value/order、
set_* 记录性 no-op。用到子集之外的,走路径 1 手工迁移(30 分钟)。

## 路径 1(垫片报错时):手工迁移决策核

## 0. 你要迁的只是"决策核",不是整个策略文件

聚宽策略 = 决策逻辑 + 平台 I/O(下单、账户、调度)。沙箱只要**决策逻辑**:
"给定截至今天的历史,明天想持有什么仓位"。下单/撮合/成本由沙箱按统一口径记账。

## 1. 契约(全部内容)

```python
class MyStrategy:
    def __init__(self, **params): ...          # 你的参数

    @staticmethod
    def param_grid():                          # ⚠️ 诚实义务:写【全部】搜索过的参数组合
        return [dict(...), ...]                #    只写最优那组 → ②③会被高估,判决无效

    def on_bar(self, ctx) -> dict:             # 每个交易日收盘时调用一次
        ...
        return {symbol: weight}                # 目标仓位:+1 满仓多 / -1 满仓空 / 0 空仓
```

`ctx` 只有三样东西(这是特性,不是限制——上下文里物理上没有未来):
- `ctx.date` — 当前交易日
- `ctx.symbols` — 当日有行情的品种
- `ctx.history(symbol, n)` — 截至**今日(含)**最近 n 个收盘价(numpy 数组,不足则短)

## 2. 聚宽 API 对照表

| 聚宽 | 沙箱 |
|---|---|
| `initialize(context)` / `g.xxx` | `__init__(self, ...)` / `self.xxx` |
| `run_daily(f)` / `handle_data` | `on_bar(self, ctx)` |
| `attribute_history(sec, n, '1d', ['close'])` | `ctx.history(symbol, n)` |
| `context.current_dt` | `ctx.date` |
| `order_target_value(sec, v)` / 仓位管理 | 直接 `return {symbol: weight}`(声明目标仓位) |
| `get_price(..., end_date=...)` | 不需要——`history` 天然截至今日,**想偷看未来也拿不到** |

## 3. 完整示例:聚宽双均线 → 沙箱

聚宽原版(节选):
```python
def initialize(context):
    g.fast, g.slow = 5, 20
def handle_data(context, data):
    hist = attribute_history(g.security, g.slow, '1d', ['close'])
    if hist['close'][-g.fast:].mean() > hist['close'].mean():
        order_target_value(g.security, context.portfolio.total_value)
    else:
        order_target_value(g.security, 0)
```

迁移后(逐行对应):
```python
import numpy as np

class DualMA:
    def __init__(self, fast=5, slow=20):
        self.fast, self.slow = fast, slow

    @staticmethod
    def param_grid():                         # 你回测时试过的全部组合,如实写
        return [dict(fast=f, slow=s) for f in (5, 10) for s in (20, 40, 60)]

    def on_bar(self, ctx):
        w = {}
        for s in ctx.symbols:
            c = ctx.history(s, self.slow)
            if len(c) < self.slow:
                w[s] = 0.0
                continue
            w[s] = 1.0 if c[-self.fast:].mean() > c.mean() else 0.0
        return w
```

## 4. 准备数据(二选一)

```python
from quantros.data import load_csv, from_akshare
data = load_csv("my_data.csv")     # 列: trading_date,symbol,close[,volume/adv]
data = from_akshare()              # 或:真实股指期货 IF/IH/IC/IM(需联网)
```
⚠️ 沙箱保证【时点诚实】,不保证【数据诚实】——你喂入的数据若含幸存者偏差/错价,
沙箱无法察觉。用交易所级或平台校验过的数据。

## 5. 跑五门判决

```python
import quantros

rets = quantros.run_sandbox_grid(DualMA, data, bps=3.0)          # ① 沙箱重放全参数族
rep  = quantros.evaluate_returns(rets, sandbox_verified=True)     # ② PBO + ③ DSR

from quantros.sandbox import as_vectorized                        # 冠军冻结成向量化接口
from quantros.regime import tail_gate                             # ④ = E2压力 + E3状态切片
from quantros.capacity import capacity_probe                      # ⑤ 容量
champ = as_vectorized(DualMA(fast=5, slow=20), data)              # 用②③选出的冠军参数
t_ok = tail_gate(champ, data, bps=3.0)
c_con, _ = capacity_probe(champ, data, target_aum=1e7, verbose=False)

quantros.final_verdict(sandbox_ok=True, pbo=rep["pbo"], dsr=rep["dsr"],
                       tail_ok=t_ok, capacity_ok=(c_con is False))
```

## 6. 当前边界(诚实清单)

- **日频、收盘价单字段**。分钟频、多字段、期权链是家族扩展(期权卖方请先走
  `quantros-gate` 通用层拿初筛,期权链沙箱在路线图上)。
- ④⑤ 探针吃向量化接口;事件式策略用 `quantros.sandbox.as_vectorized(strategy, data)`
  一行冻结接入(注意语义:冻结的是该数据上的决策序列,E2 压力测的是"既定仓位
  在压力行情下的表现")。
- **param_grid 必须是全部搜索空间**——这是平台无法代验的用户侧诚实义务;
  少报 = ③被高估 = 你在骗自己的钱。

## 7. 30 分钟检查清单

- [ ] 决策核抽出来了(不含下单/账户代码)
- [ ] `attribute_history` 全部换成 `ctx.history`
- [ ] 仓位改为 `return {symbol: weight}`
- [ ] `param_grid()` 如实列出全部试过的组合
- [ ] 数据落到规范 schema
- [ ] `run_sandbox_grid` → `evaluate_returns` → `final_verdict` 跑通
