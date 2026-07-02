# 路 B 导出指南:从你的回测平台导出 returns.csv → quantros gate

适用:任何平台(真格/QMT/聚宽/backtrader/vnpy/Excel)。策略在你自己的平台上
用真实数据跑,只导出**结果曲线**给 quantros 判——代码永不离开你的平台。

## 目标格式(二选一)

```csv
combo,date,ret          # 首选:日收益(净,扣你的真实成本后)
n5k04,2025-01-02,0.0012
n5k04,2025-01-03,-0.0008
n10k05,2025-01-02,0.0005
...
```
```csv
combo,date,equity       # 或:净值曲线(须全程>0;从0起步的累计盈亏请自行
n5k04,2025-01-02,1.0000 #   换算 ret=diff(盈亏)/初始资金——资金规模只有你知道)
```

然后:
```bash
quantros gate returns.csv --n-trials 50 --html report.html
```

## 三条诚实义务(判决可信度取决于此,平台无法代验)

1. **交全部试过的组合**——含被淘汰的差组合。只交好的 → ②被低估、③被高估,骗的是你自己的钱;
2. **`--n-trials` 报真实总尝试数**(≥提交数,含试过又放弃的);
3. **收益是净收益**(扣你实际佣金/滑点),否则③测的是毛 edge。

> 判决上限:本路径为**初筛**(未验证前视)。若你的回测本身偷看未来,
> 这里照样通过——想要可信判决走沙箱路径(MIGRATION.md)。

## 各平台导出片段

### 真格(POBO)
回测报告页导出逐日权益 CSV(或从成交记录重建),多组参数各跑一次,然后:
```python
import pandas as pd, glob
rows = []
for f in glob.glob("真格导出/*.csv"):          # 每个文件=一组参数的逐日权益
    df = pd.read_csv(f)                        # 列名按导出实际核对
    combo = f.split("/")[-1].replace(".csv", "")
    eq = df["权益"].astype(float)
    ret = eq.diff() / 500000                   # ← 换成你的初始资金
    for d, r in zip(df["日期"][1:], ret[1:]):
        rows.append({"combo": combo, "date": d, "ret": r})
pd.DataFrame(rows).to_csv("returns.csv", index=False)
```

### 聚宽(研究环境)
```python
import pandas as pd
rows = []
for combo, bt_id in {"n5k04": "回测ID1", "n10k05": "回测ID2"}.items():
    bt = get_backtest(bt_id)
    daily = bt.get_results()                   # 逐日结果,字段按账号实际核对
    prev = None
    for rec in daily:
        nav = 1 + rec["returns"]               # returns 为累计收益率
        if prev is not None:
            rows.append({"combo": combo, "date": rec["time"][:10], "ret": nav/prev - 1})
        prev = nav
pd.DataFrame(rows).to_csv("returns.csv", index=False)
```

### backtrader
```python
import backtrader as bt, pandas as pd
rows = []
for combo, params in grid.items():
    cerebro = ...                              # 你的组装
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="tr", timeframe=bt.TimeFrame.Days)
    strat = cerebro.run()[0]
    for d, r in strat.analyzers.tr.get_analysis().items():
        rows.append({"combo": combo, "date": d.date(), "ret": r})
pd.DataFrame(rows).to_csv("returns.csv", index=False)
```

### QMT / 其它 / Excel
只要凑出三列 `combo,date,ret`(或 equity)即可;Excel 手工整理后另存 CSV 同样有效。

## 只有一条曲线?

也收:`quantros gate my_curve.csv` —— ③DSR + bootstrap 照算,
② 诚实标注"单一曲线无法评估(未评估 ≠ 通过)"。想解锁 ②,补交参数族。
