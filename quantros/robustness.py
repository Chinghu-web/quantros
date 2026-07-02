"""
QuantROS 状态稳健性探针 (Robustness Probes) —— 第五根柱子。

历史只是 DGP 抽出的【一条】样本路径。一个只在那条路径上活、换条历史就死的策略是脆的。
本柱用两种互补方式检验稳健性,二者诚实度不同:

  E1 重采样稳健性 (Block Bootstrap):
    把收益率按【整块】重采样重拼,造出上千条"同分布、不同顺序"的平行历史,
    看净夏普分布的【最差分位】是否仍存活。不引入任何新假设——只问"若历史另一种展开"。
    必须用 block(而非逐点 IID)重采样,否则会打碎策略赖以为生的自相关 → 误杀真 edge。

  E2 压力注入 (Stress Injection):
    故意造数据里【没发生过】的恶劣 regime(放大波动 / 持续崩盘),看策略会不会爆。
    靠的是【明示的假设】,不是数据本身。

⚠️ 本柱的根本边界:合成数据只能检验你放进生成器的那种变化。
   E1 造不出数据里不存在的 regime(牛市历史重采样永远采不出崩盘);
   E2 的结论只在你假设的压力情景下成立。生成器的假设,就是这根柱子的边界。
"""
import datetime
import numpy as np
import polars as pl

from quantros.overfitting import _sharpe, RobustMomentumStrategy, OverfitRandomSeedStrategy
from quantros.cost import _gross_and_turnover

SYMBOLS = ["IF", "IH", "IC", "IM"]


def generate_regime_data(n_days=400, beta=0.1, drift=0.0004, seed=7, adv_dollars=5e8) -> pl.DataFrame:
    """带【正向 drift + 真实动量 edge】的价格 + ADV。
    drift 让 buy-and-hold 平时有正夏普(才能演示'平时好、崩盘死');
    beta 让动量策略有真 edge(才能演示'重采样下仍存活')。"""
    rng = np.random.default_rng(seed)
    start = datetime.date(2025, 1, 1)
    rows = []
    for sym in SYMBOLS:
        price, hist, count, i = 5000.0, [], 0, 0
        while count < n_days:
            d = start + datetime.timedelta(days=i); i += 1
            if d.weekday() >= 5: continue
            mom = float(np.mean(hist[-5:])) if len(hist) >= 5 else 0.0
            r = drift + beta * np.sign(mom) * 0.01 + rng.normal(0, 0.01)
            price *= (1 + r); hist.append(r)
            rows.append({"trading_date": d, "symbol": sym, "close": price, "adv": float(adv_dollars)})
            count += 1
    return pl.DataFrame(rows).sort(["symbol", "trading_date"])


# ── 评估:净夏普(复用成本柱的记账)──────────────────────
def _net_sharpe(strategy, df, bps):
    g, to = _gross_and_turnover(strategy, df)
    return _sharpe(g - (bps * 1e-4) * to)


# ── E1:block bootstrap 重采样稳健性 ───────────────────
def _block_bootstrap(df, L, rng):
    """block bootstrap:块内整段搬运,保住时序自相关(动量/反转)。
    按【每个品种自己的长度】重采样,以兼容参差面板(各品种上市日不同)。
    ⚠️ 参差面板下无法跨品种同步,故截面相关性不保留——对截面策略是已知边界。"""
    df = df.sort(["symbol", "trading_date"])
    has_adv = "adv" in df.columns
    frames = []
    for s in df["symbol"].unique(maintain_order=True).to_list():
        sd = df.filter(pl.col("symbol") == s)
        dates = sd["trading_date"].to_list(); T = len(dates)
        c = sd["close"].to_numpy()
        Lr = T - 1
        if Lr < 1:
            continue
        nb = int(np.ceil(Lr / L))
        starts = rng.integers(0, Lr, size=nb)
        seq = np.concatenate([(st + np.arange(L)) % Lr for st in starts])[:Lr]   # 循环块
        r = (c[1:] / c[:-1] - 1)[seq]
        new_close = c[0] * np.cumprod(np.concatenate([[1.0], 1 + r]))
        d = {"trading_date": dates, "symbol": [s] * T, "close": new_close}
        if has_adv: d["adv"] = [sd["adv"].to_numpy()[0]] * T
        frames.append(pl.DataFrame(d))
    return pl.concat(frames).sort(["symbol", "trading_date"])


def bootstrap_robustness_probe(strategy, df, n_paths=200, block=20, bps=10.0, seed=11, verbose=True):
    """E1 入口。最差 5 分位净夏普 ≤ 0 → 标记'路径脆弱'(只在真实那条历史上活)。
    ⚠️ 盲区:block 太小会毁掉 edge(误杀)、太大则只复读原路径;无法造出数据里没有的 regime。
    """
    rng = np.random.default_rng(seed)
    sh = np.array([_net_sharpe(strategy, _block_bootstrap(df, block, rng), bps) for _ in range(n_paths)])
    p5, p50 = float(np.percentile(sh, 5)), float(np.percentile(sh, 50))
    frac_pos = float(np.mean(sh > 0))
    # 稳健 = 多数平行历史为正 且 中位为正;不用 p5 单点判,避免偶发破 0 误杀
    fragile = bool(frac_pos < 0.70 or p50 <= 0)
    # 为正占比的二项标准误;判据贴近 0.70 阈值(2 倍 SE 内)则置信不足
    se = float(np.sqrt(frac_pos * (1 - frac_pos) / n_paths))
    conf = " ⚠贴近阈值,置信不足" if abs(frac_pos - 0.70) < 2 * se else ""
    detail = (f"平行历史 net夏普 中位={p50:.2f} 最差5%={p5:.2f} "
              f"为正占比={frac_pos:.0%}(n={n_paths},SE±{se:.0%}){conf}")
    if verbose:
        print(f" [E1 重采样稳健性] {'⚠️ 路径脆弱' if fragile else '✓ 通过'} | {detail}")
    return fragile, detail


# ── E2:压力注入 ───────────────────────────────────────
def _vol_scale(df, factor):
    """放大波动:保留每品种 drift/edge,把偏离均值的部分 ×factor。"""
    frames = []
    has_adv = "adv" in df.columns
    for s in df["symbol"].unique(maintain_order=True).to_list():
        sd = df.sort("trading_date").filter(pl.col("symbol") == s)
        c = sd["close"].to_numpy(); r = c[1:] / c[:-1] - 1
        r2 = r.mean() + factor * (r - r.mean())
        new_close = c[0] * np.cumprod(np.concatenate([[1.0], 1 + r2]))
        d = {"trading_date": sd["trading_date"].to_list(), "symbol": [s] * len(c), "close": new_close}
        if has_adv: d["adv"] = [sd["adv"].to_numpy()[0]] * len(c)
        frames.append(pl.DataFrame(d))
    return pl.concat(frames).sort(["symbol", "trading_date"])


def _inject_crash(df, daily=-0.03, length=20, at=0.7):
    """持续崩盘:在 at 处把所有品种连续 length 天的收益设为 daily(同步)。
    持续下跌(非 V 形)→ buy-and-hold 巨亏,动量策略应翻空避险。"""
    frames = []
    has_adv = "adv" in df.columns
    for s in df["symbol"].unique(maintain_order=True).to_list():
        sd = df.sort("trading_date").filter(pl.col("symbol") == s)
        c = sd["close"].to_numpy(); r = c[1:] / c[:-1] - 1
        k = int(len(r) * at)
        r[k:k + length] = daily
        new_close = c[0] * np.cumprod(np.concatenate([[1.0], 1 + r]))
        d = {"trading_date": sd["trading_date"].to_list(), "symbol": [s] * len(c), "close": new_close}
        if has_adv: d["adv"] = [sd["adv"].to_numpy()[0]] * len(c)
        frames.append(pl.DataFrame(d))
    return pl.concat(frames).sort(["symbol", "trading_date"])


def stress_probe(strategy, df, bps=10.0, verbose=True):
    """E2 入口。任一压力情景下净夏普 < -0.5 → 标记'压力脆弱'(扛不住没见过的恶劣 regime)。
    ⚠️ 盲区:结论只在所选压力情景下成立;情景由人设定,非数据自带。
    """
    scenarios = {"波动×2": _vol_scale(df, 2.0), "持续崩盘": _inject_crash(df)}
    res = {k: round(_net_sharpe(strategy, v, bps), 2) for k, v in scenarios.items()}
    worst = min(res.values())
    fragile = bool(worst < -0.5)
    if verbose:
        print(f" [E2 压力注入] {'⚠️ 压力脆弱' if fragile else '✓ 通过'} | 各情景 net夏普={res}")
    return fragile, f"各情景 net夏普={res}"


# ── 稳健性动物园 ───────────────────────────────────────
class AlwaysLongStrategy:
    """野兽:永远满仓做多。平时吃 drift(E1 下也稳),但持续崩盘里巨亏 → E2 拦截。"""
    def generate_signals(self, df):
        return df.with_columns(pl.lit(1.0).alias("weight"))
