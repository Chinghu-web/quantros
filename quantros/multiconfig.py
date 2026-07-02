"""
QuantROS 中档诊断 (Multi-config Diagnosis) —— outputs-only 获客钩子。

用户不交源码,只交【多个参数配置各自的持仓序列】+ 价格。
关键洞察:PBO(回测过拟合概率)只需要"各配置的收益矩阵",不需要策略代码——
所以不暴露逻辑也能做金标准过拟合检测。成本/容量本就只需持仓+价格,满血。

能力(相对完整五柱):
  ✅ 二·过拟合 PBO   —— 从持仓恢复收益矩阵即可,无需重跑代码
  ✅ 三·成本         —— 满血
  ✅ 四·容量         —— 满血(需 ADV 列)
  ⚠️ 一·因果 / 五·稳健 —— 需重跑策略,本档做不了(完整模式才有)

连接接缝:diagnose_multiconfig(..., report_sink=fn)。
report_sink 只收到【判决+指标】的 dict,绝不含代码/原始持仓——这就是"连到你系统"的口。
"""
import numpy as np
import polars as pl

from quantros.overfitting import pbo_cscv
from quantros.cost import breakeven_bps, _gross_and_turnover
from quantros.capacity import capacity_probe


class _FrozenStrategy:
    """把一组【给定持仓】包成策略接口,从而无缝复用成本/容量探针(它们要 .generate_signals)。"""
    def __init__(self, weights): self._w = weights
    def generate_signals(self, df):
        return df.with_columns(pl.Series("weight", self._w))


def _align_weights(positions, prices) -> np.ndarray:
    """把 (trading_date, symbol, weight) 持仓按 prices 的 [symbol,trading_date] 排序对齐成数组。
    与成本/容量探针内部的排序一致,保证 _FrozenStrategy 喂出的权重对得上行。"""
    j = (prices.sort(["symbol", "trading_date"])
               .join(positions, on=["trading_date", "symbol"], how="left"))
    return j["weight"].to_numpy()


def build_return_matrix(positions_by_config, prices):
    """各配置 → 按交易日聚合的组合收益,拼成矩阵 M[t, j]。返回 (M, dates, names)。
    weight[t] 实现 t→t+1 远期收益(shift(-1) 仅用于记账)。"""
    prices = prices.sort(["symbol", "trading_date"])
    base = prices.with_columns(
        ((pl.col("close").shift(-1).over("symbol") / pl.col("close")) - 1).alias("_fr"))
    cols, names, dates = [], [], None
    for name, pos in positions_by_config.items():
        w = _align_weights(pos, prices)
        tmp = base.with_columns(pl.Series("_w", w))
        tmp = tmp.with_columns((pl.col("_w") * pl.col("_fr")).alias("_pnl"))
        daily = (tmp.group_by("trading_date").agg(pl.col("_pnl").mean().alias("p"))
                    .sort("trading_date"))
        if dates is None:
            dates = daily["trading_date"].to_numpy()
        cols.append(daily["p"].to_numpy()); names.append(name)
    M = np.column_stack(cols)
    good = ~np.isnan(M).any(axis=1)
    return M[good], dates[good], names


def diagnose_multiconfig(positions_by_config, prices, *, realistic_bps=10.0,
                         target_aum=1e8, k=0.01, S=10, pbo_threshold=0.30,
                         report_sink=None, verbose=True):
    """中档诊断:从【多配置持仓 + 价格】出 PBO + 逐配置成本/容量,无需源码。

    report_sink: 可选回调,收到不含代码/原始持仓的【判决 dict】——连云端的接缝。
    返回判决 dict。
    """
    if len(positions_by_config) < 2:
        raise ValueError("中档 PBO 至少需要 2 个参数配置的持仓")
    M, _, names = build_return_matrix(positions_by_config, prices)
    pbo, _ = pbo_cscv(M, S=S)
    has_adv = "adv" in prices.columns

    per_config = {}
    for name in names:
        w = _align_weights(positions_by_config[name], prices)
        fs = _FrozenStrategy(w)
        be = breakeven_bps(fs, prices)
        cost_fail = bool(be < realistic_bps)
        entry = {"breakeven_bps": round(be, 2), "cost_fail": cost_fail}
        if has_adv:
            constrained, cap = capacity_probe(fs, prices, k=k, target_aum=target_aum, verbose=False)
            cap_clean = None if (cap is None or cap != cap) else cap   # nan/None → JSON 安全
            entry.update({"capacity": cap_clean, "capacity_fail": bool(constrained)})
        per_config[name] = entry

    overfit = bool(pbo > pbo_threshold)
    cost_fails = [n for n, e in per_config.items() if e["cost_fail"]]
    cap_fails = [n for n, e in per_config.items() if e.get("capacity_fail")]
    report = {
        "mode": "multi-config (outputs-only)",
        "n_configs": len(names),
        "pbo": round(pbo, 3),
        "overfit_risk": overfit,
        "cost_fail_configs": cost_fails,
        "capacity_fail_configs": cap_fails,
        "per_config": per_config,
        "not_assessed": ["一·因果", "五·稳健"],   # 诚实:本档够不到的维度
    }

    if verbose:
        print("=" * 64)
        print(f" QuantROS 中档诊断(outputs-only,{len(names)} 个配置)")
        print("-" * 64)
        print(f"  二·过拟合 PBO   {'❌ 高风险' if overfit else '✅ 通过'}   PBO={pbo:.3f} (阈值 {pbo_threshold})")
        print(f"  三·成本         {'❌ ' + str(len(cost_fails)) + ' 个配置不达标' if cost_fails else '✅ 全部达标'}")
        if has_adv:
            print(f"  四·容量         {'❌ ' + str(len(cap_fails)) + ' 个配置受限' if cap_fails else '✅ 全部达标'}")
        else:
            print("  四·容量         ⚪ BLIND(价格数据缺 ADV 列)")
        print("  一·因果/五·稳健  ⚪ 未评估(需完整模式重跑策略,未评估 ≠ 通过)")
        print("=" * 64)

    if report_sink is not None:
        report_sink(report)        # 连接点:只把判决送出去,代码/原始持仓永不离开本地
    return report


# ── 本地 CLI 入口:读 CSV → 出诊断 ─────────────────────
def load_from_csv(positions_csv, prices_csv):
    """positions_csv 列: config, trading_date, symbol, weight
       prices_csv    列: trading_date, symbol, close [, adv]
    返回 (positions_by_config, prices),可直接喂给 diagnose_multiconfig。"""
    pos = pl.read_csv(positions_csv, try_parse_dates=True)
    prices = pl.read_csv(prices_csv, try_parse_dates=True)
    by_config = {name: g.drop("config")
                 for name, g in pos.group_by("config", maintain_order=True)}
    by_config = {(n[0] if isinstance(n, tuple) else n): df for n, df in by_config.items()}
    return by_config, prices


def main(argv=None):
    import argparse, json
    p = argparse.ArgumentParser(prog="quantros-multiconfig",
                                description="QuantROS 中档诊断(只需多配置持仓+价格,无需源码)")
    p.add_argument("positions_csv", help="config,trading_date,symbol,weight")
    p.add_argument("prices_csv", help="trading_date,symbol,close[,adv]")
    p.add_argument("--realistic-bps", type=float, default=10.0)
    p.add_argument("--target-aum", type=float, default=1e8)
    p.add_argument("--out", help="把判决 JSON 写到此文件(本地;不联网)")
    a = p.parse_args(argv)
    by_config, prices = load_from_csv(a.positions_csv, a.prices_csv)
    report = diagnose_multiconfig(by_config, prices, realistic_bps=a.realistic_bps,
                                  target_aum=a.target_aum, verbose=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        print(f"判决已写入 {a.out}(仅指标,无代码/原始持仓)")
    return report


if __name__ == "__main__":
    main()
