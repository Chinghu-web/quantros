"""
QuantROS 过拟合探针 (Overfitting Probes) —— 第二根柱子。

因果性探针回答"策略有没有偷看未来";过拟合探针回答另一个问题:
"这条漂亮的回测曲线,是真 edge,还是在样本里调参调出来的运气?"

两个探针,各有诚实边界:
  O1 样本外退化 (Walk-forward OOS) —— 样本内选最优参数,样本外看夏普崩不崩。
  O2 回测过拟合概率 PBO (CSCV) —— Bailey & López de Prado 的组合对称交叉验证,
     判断"样本内最优配置"在样本外是否系统性地跌到中位数以下。

哲学与因果性探针一致:不审代码,审行为;每个探针都标注它对什么失明。
"""
import datetime
from itertools import combinations
import numpy as np
import polars as pl


# ── 带真实 edge 的数据 ─────────────────────────────────
def generate_data_with_edge(n_days=400, beta=0.30, seed=7) -> pl.DataFrame:
    """生成埋有【真实持续动量 edge】的价格序列。

    在纯随机游走上,任何样本内最优策略都必然过拟合(这本身是探针的正确结论)。
    要证明探针能【区分】真假,必须埋一个跨样本稳定的真信号:
    下一日收益向过去动量的方向轻微倾斜(beta),叠加噪声。
    用动量的策略因此有真 edge;用随机信号的策略只有运气。
    """
    symbols = ["IF", "IH", "IC", "IM"]
    rng = np.random.default_rng(seed)
    start = datetime.date(2025, 1, 1)
    rows = []
    for sym in symbols:
        price = 5000.0
        hist = []
        count, i = 0, 0
        while count < n_days:
            d = start + datetime.timedelta(days=i); i += 1
            if d.weekday() >= 5: continue
            mom = float(np.mean(hist[-5:])) if len(hist) >= 5 else 0.0
            r = beta * np.sign(mom) * 0.01 + rng.normal(0, 0.01)   # 真 edge + 噪声
            price *= (1 + r)
            hist.append(r)
            rows.append({"trading_date": d, "symbol": sym, "close": price})
            count += 1
    return pl.DataFrame(rows).sort(["symbol", "trading_date"])


# ── 诚实回测核心(只做记账,不产生信号)──────────────────
def _sharpe(r: np.ndarray) -> float:
    r = r[~np.isnan(r)]
    if len(r) < 2 or r.std() == 0: return 0.0
    return float(r.mean() / r.std() * np.sqrt(252))


def config_return_matrix(strategy_cls, df):
    """对策略的整个参数族跑回测,返回 (M, dates, configs)。
    M[t, j] = 第 j 个参数配置在第 t 个交易日的组合收益。
    持仓 weight[t] 实现的是 t→t+1 的远期收益(shift(-1) 仅用于记账 PnL,
    不进入信号通路,因此不构成前视——这点与因果性柱子衔接)。
    """
    configs = strategy_cls.param_grid() if hasattr(strategy_cls, "param_grid") else [{}]
    df = df.sort(["symbol", "trading_date"])
    base = df.with_columns(
        ((pl.col("close").shift(-1).over("symbol") / pl.col("close")) - 1).alias("_fr"))
    cols, dates = [], None
    for cfg in configs:
        w = strategy_cls(**cfg).generate_signals(df)["weight"].to_numpy()
        tmp = base.with_columns(pl.Series("_w", w))
        tmp = tmp.with_columns((pl.col("_w") * pl.col("_fr")).alias("_pnl"))
        daily = tmp.group_by("trading_date").agg(pl.col("_pnl").mean().alias("p")).sort("trading_date")
        if dates is None: dates = daily["trading_date"].to_numpy()
        cols.append(daily["p"].to_numpy())
    M = np.column_stack(cols)
    good = ~np.isnan(M).any(axis=1)            # 丢掉每段末尾远期收益为 NaN 的行
    return M[good], dates[good], configs


# ── 探针 O1:样本外退化 ─────────────────────────────────
def oos_degradation_probe(strategy_cls, df, split=0.5, verbose=True):
    """样本内挑最优参数 → 样本外检验。模拟真实过拟合陷阱:在历史上调参,上线后崩。

    判定:样本内夏普像样(>0.5)但样本外跌破 max(0, 0.3×样本内) → 标记过拟合。
    返回 (is_overfit, detail)。
    ⚠️ 盲区:单切分有噪声,且无法区分"过拟合"与"市场状态切换";需足够数据。
    """
    M, _, configs = config_return_matrix(strategy_cls, df)
    T = M.shape[0]; k = int(T * split)
    n_oos = T - k
    is_sh = np.array([_sharpe(M[:k, j]) for j in range(M.shape[1])])
    oos_sh = np.array([_sharpe(M[k:, j]) for j in range(M.shape[1])])
    j = int(np.argmax(is_sh))
    overfit = bool(is_sh[j] > 0.5 and oos_sh[j] < max(0.0, 0.3 * is_sh[j]))
    # 样本外夏普的 t 值 = 年化夏普 × √(样本外年数);|t|<2 即统计上分辨不出与 0 的差别
    t_oos = oos_sh[j] * np.sqrt(n_oos / 252.0)
    conf = "" if abs(t_oos) >= 2 else f" ⚠样本外样本不足(t={t_oos:.1f},n={n_oos})"
    detail = (f"最优配置={configs[j]} 样本内夏普={is_sh[j]:.2f} "
              f"样本外夏普={oos_sh[j]:.2f}(t={t_oos:.1f}){conf}")
    if verbose:
        print(f" [O1 样本外退化] {'⚠️ 疑似过拟合' if overfit else '✓ 通过'} | {detail}")
    return overfit, detail


# ── 探针 O2:回测过拟合概率 PBO (CSCV) ──────────────────
def pbo_cscv(M, S=10):
    """组合对称交叉验证 → 回测过拟合概率 (Bailey & López de Prado, 2014)。

    把时间轴切成 S 段,枚举所有"一半做样本内、另一半做样本外"的组合;
    每个组合里取样本内夏普最高的配置,看它在样本外的相对排名。
    若样本内冠军在样本外系统性地落到中位数以下(logit λ<0),即为过拟合证据。
    PBO = λ<0 的组合占比。返回 (pbo, lambdas)。
    """
    T, N = M.shape
    S -= S % 2                                  # S 取偶数
    cut = T - (T % S); M = M[:cut]
    chunks = np.array_split(np.arange(cut), S)
    lambdas = []
    for combo in combinations(range(S), S // 2):
        is_idx = np.concatenate([chunks[c] for c in combo])
        oos_idx = np.concatenate([chunks[c] for c in range(S) if c not in combo])
        is_sh = np.array([_sharpe(M[is_idx, j]) for j in range(N)])
        oos_sh = np.array([_sharpe(M[oos_idx, j]) for j in range(N)])
        j = int(np.argmax(is_sh))               # 样本内冠军
        rank = int(np.sum(oos_sh < oos_sh[j]) + 1)   # 它在样本外的排名(越高越好)
        w = min(max(rank / (N + 1), 1e-6), 1 - 1e-6)
        lambdas.append(np.log(w / (1 - w)))
    lambdas = np.array(lambdas)
    return float(np.mean(lambdas < 0)), lambdas


def pbo_probe(strategy_cls, df, S=10, threshold=0.30, verbose=True):
    """PBO 探针入口。pbo>threshold → 标记过拟合倾向。
    ⚠️ 盲区:需要一个【参数族】才能评估;对单个手调、无 param_grid 的策略失明
    (与因果性柱子'硬编码常数失明'同理——没有可观测的搜索空间就无法物理评估)。
    """
    M, _, configs = config_return_matrix(strategy_cls, df)
    if len(configs) < 2:
        if verbose: print(" [O2 PBO] ∅ 无参数族,探针对此失明")
        return None, 1.0
    pbo, _ = pbo_cscv(M, S=S)
    overfit = pbo > threshold
    if verbose:
        print(f" [O2 PBO] {'⚠️ 过拟合概率高' if overfit else '✓ 通过'} | PBO={pbo:.2f} (配置数={len(configs)})")
    return overfit, pbo


# ── 过拟合动物园 ───────────────────────────────────────
class RobustMomentumStrategy:
    """诚实:用过去收益的滚动均值做动量,吃数据里埋的真 edge。
    所有合理回看窗口都为正(参数高原)→ 样本外稳定,PBO 低。应放行。"""
    def __init__(self, lookback=5): self.lookback = lookback
    @staticmethod
    def param_grid(): return [dict(lookback=k) for k in (3, 5, 10, 20)]
    def generate_signals(self, df):
        df = df.with_columns(pl.col("close").pct_change().over("symbol").alias("r"))
        df = df.with_columns(pl.col("r").rolling_mean(self.lookback).over("symbol").alias("m"))
        return df.with_columns(pl.when(pl.col("m") > 0).then(1.0).otherwise(-1.0).alias("weight"))


class OverfitRandomSeedStrategy:
    """过拟合野兽:每个 seed 是一组固定随机多空,无任何真结构。
    样本内冠军纯靠运气,样本外坍塌 → O1 抓退化、O2 给出高 PBO(≈0.5)。"""
    def __init__(self, seed=0): self.seed = seed
    @staticmethod
    def param_grid(): return [dict(seed=s) for s in range(40)]
    def generate_signals(self, df):
        w = np.random.default_rng(self.seed).choice([-1.0, 1.0], size=len(df))
        return df.with_columns(pl.Series("weight", w))
