"""
QuantROS 探针 E3 · 历史状态切片 (Regime Slice) —— 把"在不同市况下都稳住"变成可证伪的门。

用户直觉:"整体设计合理,在过去各种市场状态下都稳住不崩,下一个环境就能跑。"
诚实版本:【在所有已见状态中未被证伪】是必要条件,不是充分条件——
  · E3 只能检验历史里【出现过】的状态(牛/熊/高波/低波);
  · "下一个未知状态"在逻辑上不可归纳(火鸡问题);未见状态只能靠 E2 压力注入
    去测【假设的】情景,谁也不能承诺未知。
  · 尤其警惕:空波动率/卖权利金类策略,"历史各状态全稳"正是它爆掉前的标准长相
    ——只要样本里没出现过 vol spike,E3 必然全绿。E3 全绿 ≠ 尾部安全。

方法:市场状态 = 滚动波动(高/低,按中位数分)× 滚动趋势(上/下),逐日打标;
策略净收益按状态归组,任一状态(样本 ≥ min_days)累计亏损 < -blowup_dd → 证伪("崩")。
判据用【状态内累计亏损】而非夏普:崩 = 集中在某一市况里的深亏,这才是"设计押错市况"。
"""
import numpy as np
import polars as pl

from quantros.overfitting import _sharpe
from quantros.robustness import stress_probe


def _net_by_date(strategy, df, bps):
    """净日收益 + 对齐日期(口径与成本柱一致:t 仓位吃 t→t+1,净 = 毛 − bps×换手)。"""
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
    agg = (tmp.group_by("trading_date")
              .agg([pl.col("_g").mean().alias("g"), pl.col("_to").mean().alias("to")])
              .sort("trading_date"))
    g = agg["g"].to_numpy(); to = agg["to"].to_numpy()
    dates = agg["trading_date"].to_numpy()
    net = g - (bps * 1e-4) * to
    good = ~np.isnan(net)
    return dates[good], net[good]


def _market_states(df, window=20):
    """逐日市场状态标签:滚动波动(高/低)× 滚动趋势(上/下)。暖机期标 None。"""
    mkt = (df.sort(["symbol", "trading_date"])
             .with_columns(pl.col("close").pct_change().over("symbol").alias("_r"))
             .group_by("trading_date").agg(pl.col("_r").mean().alias("mr"))
             .sort("trading_date"))
    mkt = mkt.with_columns([
        pl.col("mr").rolling_std(window).alias("vol"),
        pl.col("mr").rolling_mean(window).alias("tr"),
    ])
    vol = mkt["vol"].to_numpy(); tr = mkt["tr"].to_numpy()
    med = float(np.nanmedian(vol))
    labels = {}
    for d, v, t in zip(mkt["trading_date"].to_numpy(), vol, tr):
        if np.isnan(v) or np.isnan(t):
            labels[d] = None
        else:
            labels[d] = f"{'上行' if t > 0 else '下行'}/{'高波' if v > med else '低波'}"
    return labels


def regime_slice_probe(strategy, df, bps=10.0, min_days=10, blowup_dd=0.20, verbose=True):
    """E3 入口。任一已见状态内累计亏损超过 blowup_dd → 证伪("押错市况会崩")。
    返回 (blown, detail)。
    ⚠️ 归纳边界(记录在案):E3 只检验历史里出现过的状态;对没出现过的 regime
    结构性失明(有分工断言为证:平静史上的满仓多 E3 放行、E2 拦截)。"""
    dates, net = _net_by_date(strategy, df, bps)
    labels = _market_states(df)
    by_state = {}
    for d, r in zip(dates, net):
        s = labels.get(d)
        if s is None: continue
        by_state.setdefault(s, []).append(r)

    rows, blown, blown_states = [], False, []
    for s in sorted(by_state):
        r = np.asarray(by_state[s])
        if len(r) < min_days:
            rows.append(f"{s}: 样本不足({len(r)}天,不判)")
            continue
        cum = float(np.prod(1 + r) - 1)
        sh = _sharpe(r)
        bad = cum < -blowup_dd
        if bad: blown, blown_states = True, blown_states + [s]
        rows.append(f"{s}: {len(r)}天 累计={cum:+.1%} 夏普={sh:.2f}{' ←崩' if bad else ''}")
    detail = " | ".join(rows)
    if verbose:
        print(f" [E3 状态切片] {'⚠️ 状态崩溃:' + ','.join(blown_states) if blown else '✓ 已见状态全稳'}")
        for line in rows: print(f"     {line}")
        if not blown:
            print("     (⚠ 已见状态全稳 ≠ 未知状态安全——归纳边界,未见 regime 由 E2 压力测)")
    return blown, detail


def tail_gate(strategy, df, bps=10.0):
    """风险门④的便捷组合:E2 压力注入(假设的未见情景) + E3 状态切片(已见历史)。
    两者都不崩 → tail_ok=True,可直接喂 verdict.final_verdict(tail_ok=...)。"""
    e2, _ = stress_probe(strategy, df, bps=bps, verbose=False)
    e3, _ = regime_slice_probe(strategy, df, bps=bps, verbose=False)
    return (not e2) and (not e3)


# ── 状态动物园 ─────────────────────────────────────────
class MartingaleDipStrategy:
    """野兽:平时满仓多,回撤后加倍摊平(散户经典)。上行史里曲线极稳,
    进入持续下跌状态即深亏 → 应被 E3 在'下行/高波'状态拦截。"""
    def generate_signals(self, df):
        df = df.sort(["symbol", "trading_date"])
        df = df.with_columns(pl.col("close").pct_change().over("symbol").alias("_r"))
        df = df.with_columns(pl.col("_r").rolling_sum(5).over("symbol").alias("_c5"))
        return df.with_columns(
            pl.when(pl.col("_c5") < -0.03).then(3.0).otherwise(1.0).alias("weight"))
