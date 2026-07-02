"""
QuantROS 成本敏感性探针 (Cost Probes) —— 第三根柱子。

前两根柱子放行的策略,可能因果干净、不过拟合,却仍死于现实:
换手太高、每笔 edge 太薄,扣掉手续费 + 滑点就归零。这是散户回测最常见的自欺
(回测里压根没算成本)。

探针 C1 盈亏平衡成本 (Breakeven Cost):
  组合净收益对成本率是【线性】的 ——
      净均值日收益 = 毛均值日收益 − 成本率 × 平均日换手
  令其为 0,可解析解出盈亏平衡成本率 = 毛均值 / 平均换手。
  这是"每单位换手要付多少 bp 手续费才会让策略归零"的物理刻度。
  若它低于该品种的真实成本(手续费 + 滑点),策略上线即负 → 标记脆弱。

哲学一致:不审代码,审行为;探针标注它对什么失明。
"""
import numpy as np
import polars as pl

from quantros.overfitting import _sharpe


def _gross_and_turnover(strategy, df):
    """返回按交易日聚合的 (毛日收益, 日换手) 两条等长序列。

    换手[t] = |w[t] − w[t−1]|(每品种首仓视为从 0 建仓);
    毛收益[t] = w[t] × 远期收益(t→t+1)。两者都按品种取均值再并到交易日,
    口径一致,保证 净 = 毛 − 成本率 × 换手 成立。
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
    agg = (tmp.group_by("trading_date")
              .agg([pl.col("_g").mean().alias("g"), pl.col("_to").mean().alias("to")])
              .sort("trading_date"))
    g = agg["g"].to_numpy(); to = agg["to"].to_numpy()
    good = ~np.isnan(g)                         # 丢掉末尾远期收益为 NaN 的行
    return g[good], to[good]


def breakeven_bps(strategy, df) -> float:
    """盈亏平衡成本(bp / 单位换手)。低 = 脆弱;∞ = 无换手或毛收益非正时退化。"""
    g, to = _gross_and_turnover(strategy, df)
    mean_to = to.mean()
    if mean_to <= 1e-12:                        # 几乎不换手
        return float("inf")
    return float(g.mean() / mean_to * 1e4)


def _net_sharpe(g, to, bps) -> float:
    return _sharpe(g - (bps * 1e-4) * to)


def cost_sensitivity_probe(strategy, df, realistic_bps=10.0, verbose=True):
    """C1 入口。breakeven < 真实成本 → 标记'成本脆弱'(上线即负)。

    返回 (is_fragile, detail)。同时报出年化换手与净夏普 vs 成本扫描曲线。
    ⚠️ 盲区:真实成本随品种/规模/时段变化,realistic_bps 需按标的设定;
    本探针不建模冲击成本(资金做大后的容量问题属第四根柱子,待建)。
    """
    g, to = _gross_and_turnover(strategy, df)
    be = breakeven_bps(strategy, df)
    ann_turnover = float(to.mean() * 252)
    sweep = {b: round(_net_sharpe(g, to, b), 2) for b in (0, 1, 2, 5, 10)}
    fragile = bool(be < realistic_bps)
    detail = (f"盈亏平衡={be:.2f}bp 真实成本={realistic_bps}bp "
              f"年化换手={ann_turnover:.0f}x 净夏普@{realistic_bps}bp={_net_sharpe(g, to, realistic_bps):.2f}")
    if verbose:
        print(f" [C1 成本敏感性] {'⚠️ 成本脆弱' if fragile else '✓ 通过'} | {detail}")
        print(f"     净夏普 vs 成本(bp): {sweep}")
    return fragile, detail


# ── 成本动物园 ─────────────────────────────────────────
class LowTurnoverMomentumStrategy:
    """诚实:用较长回看窗口的动量,持仓平滑、换手低 → 扛得住真实成本。应放行。"""
    def __init__(self, lookback=20): self.lookback = lookback
    def generate_signals(self, df):
        df = df.with_columns(pl.col("close").pct_change().over("symbol").alias("r"))
        df = df.with_columns(pl.col("r").rolling_mean(self.lookback).over("symbol").alias("m"))
        return df.with_columns(pl.when(pl.col("m") > 0).then(1.0).otherwise(-1.0).alias("weight"))


class ChurnyMomentumStrategy:
    """成本野兽:用当日收益符号做信号,edge 是真的但每日翻仓,换手极高。
    毛收益为正,净收益被手续费吃光 → 盈亏平衡成本极低,真实成本下翻负。"""
    def generate_signals(self, df):
        df = df.with_columns(pl.col("close").pct_change().over("symbol").alias("r"))
        return df.with_columns(pl.when(pl.col("r") > 0).then(1.0).otherwise(-1.0).alias("weight"))
