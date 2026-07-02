"""
QuantROS 容量探针 (Capacity Probe) —— 第四根柱子。

前三根柱子放行的策略,可能因果干净、不过拟合、扛得住固定手续费,
却仍死于规模:资金做大后,自己的单子冲击市场,冲击成本随规模上升,
最终把 edge 吃光。同样的信号,在流动性好的品种能装几个亿,在小品种几百万即自杀。

模型(平方根冲击律,Almgren-Chriss 业界标准):
  参与率 p = 单日下单额 / 日均成交额(ADV)
  冲击成本率 = k·√p   →   单日冲击拖累 ∝ √AUM
  故 净均日收益(AUM) = 毛均日收益 − A·√AUM,令其为 0 即解出【容量】。

⚠️ 与前三根柱子的本质区别:因果性是物理可证的,容量【只是模型估计】——
依赖冲击系数 k 与 √ 律假设。本探针给出的是数量级参考,不是物理真值。
"""
import datetime
import numpy as np
import polars as pl

from quantros.overfitting import _sharpe

SYMBOLS = ["IF", "IH", "IC", "IM"]


def generate_capacity_data(adv_dollars=5e8, beta=0.05, seed=7) -> pl.DataFrame:
    """带真实动量 edge 的价格 + 每品种日均成交额(ADV,美元)。
    adv_dollars 越小越不流动 → 容量越低。"""
    rng = np.random.default_rng(seed)
    start = datetime.date(2025, 1, 1)
    rows = []
    for sym in SYMBOLS:
        price, hist, count, i = 5000.0, [], 0, 0
        while count < 400:
            d = start + datetime.timedelta(days=i); i += 1
            if d.weekday() >= 5: continue
            mom = float(np.mean(hist[-5:])) if len(hist) >= 5 else 0.0
            r = beta * np.sign(mom) * 0.01 + rng.normal(0, 0.01)
            price *= (1 + r); hist.append(r)
            rows.append({"trading_date": d, "symbol": sym, "close": price, "adv": float(adv_dollars)})
            count += 1
    return pl.DataFrame(rows).sort(["symbol", "trading_date"])


def _daily_gross_and_impact_base(strategy, df):
    """返回 (毛日收益序列, 冲击基序列 S_t)。
    S_t = Σ_i |Δw_i|^1.5 / √(ADV_i),单日冲击拖累 = k·√(AUM/N)/N · S_t。
    """
    df = df.sort(["symbol", "trading_date"])
    base = df.with_columns(
        ((pl.col("close").shift(-1).over("symbol") / pl.col("close")) - 1).alias("_fr"))
    w = strategy.generate_signals(df)["weight"].to_numpy()
    tmp = base.with_columns(pl.Series("_w", w))
    tmp = tmp.with_columns([
        (pl.col("_w") * pl.col("_fr")).alias("_g"),
        (pl.col("_w") - pl.col("_w").shift(1).over("symbol"))
            .abs().fill_null(pl.col("_w").abs()).alias("_to"),
    ])
    tmp = tmp.with_columns((pl.col("_to").pow(1.5) / pl.col("adv").sqrt()).alias("_s"))
    agg = (tmp.group_by("trading_date")
              .agg([pl.col("_g").mean().alias("g"), pl.col("_s").sum().alias("s")])
              .sort("trading_date"))
    g = agg["g"].to_numpy(); s = agg["s"].to_numpy()
    good = ~np.isnan(g)
    return g[good], s[good]


def _net_series(g, s, aum, k, n):
    drag = k * np.sqrt(aum / n) / n * s
    return g - drag


def capacity_probe(strategy, df, k=0.01, target_aum=1e8, verbose=True):
    """容量探针入口。容量 < 目标资金规模 → 标记'规模脆弱'。

    容量 = 毛均日收益归零时的 AUM = N·(毛均·N / (k·S均))²。
    返回 (is_constrained, capacity_dollars):
        False/容量 → 通过;True/容量 → 规模脆弱;
        None/nan  → N/A(有交易但无毛 edge,容量无意义,绝不当成"通过")。
    ⚠️ 盲区:容量是【模型估计】,随冲击系数 k 与 √ 律假设变化;
    不交易的策略容量=∞(无冲击);不建模日内拆单 / 多日建仓。
    """
    g, s = _daily_gross_and_impact_base(strategy, df)
    n = df["symbol"].n_unique()
    gross_mean, s_mean = g.mean(), s.mean()
    if s_mean <= 1e-18:
        capacity, constrained = float("inf"), False        # 不交易 → 无冲击约束
    elif gross_mean <= 0:
        if verbose:
            print(" [D1 容量] ⚪ N/A(有交易但无毛 edge,容量无意义——未评估 ≠ 通过)")
        return None, float("nan")                          # 无 edge 可放大 → 容量谈不上
    else:
        capacity = float(n * (gross_mean * n / (k * s_mean)) ** 2)
        constrained = bool(capacity < target_aum)
    sweep = {f"{a:.0e}": round(_sharpe(_net_series(g, s, a, k, n)), 2)
             for a in (1e6, 1e7, 1e8, 1e9)}
    if verbose:
        print(f" [D1 容量] {'⚠️ 规模脆弱' if constrained else '✓ 通过'} | "
              f"容量≈${capacity:.2e} 目标=${target_aum:.0e}")
        print(f"     净夏普 vs 规模($): {sweep}")
    return constrained, capacity
