"""
探针 C：跨市场泛化测试 (Generalization Probe) —— 启发式警告，非物理证明。
针对动态扰动抓不到的"硬编码未来常数"作弊。
原理：硬编码常数是针对某份特定数据的量级调出来的。把价格【整体缩放】到
另一个量级（保持相对结构/基差比例不变），真策略的信号方向分布应基本稳定，
而硬编码绝对阈值的策略会因为量级变了而信号坍缩。
"""
import numpy as np
import polars as pl


def generalization_probe(strategy, df_original, df_rescaled, verbose=True):
    wo = strategy.generate_signals(df_original.sort(["symbol","trading_date"]))["weight"].to_numpy()
    wr = strategy.generate_signals(df_rescaled.sort(["symbol","trading_date"]))["weight"].to_numpy()
    m = (~np.isnan(wo)) & (~np.isnan(wr))
    if m.sum() < 10:
        return False, "有效信号太少"
    wo, wr = wo[m], wr[m]
    frac_o, frac_r = np.mean(wo > 0), np.mean(wr > 0)
    # 只用强信号：rescaled 市场里信号坍缩到 >97% 单方向，且原市场并未坍缩
    collapse_r = (frac_r > 0.97 or frac_r < 0.03)
    orig_balanced = (0.1 < frac_o < 0.9)
    suspicious = collapse_r and orig_balanced
    reason = f"原市场做多占比={frac_o:.2f}, 缩放市场做多占比={frac_r:.2f}"
    if verbose:
        print(f" [泛化探针] {'⚠️ 疑似硬编码' if suspicious else '✓ 通过'} | {reason}")
    return suspicious, reason


def make_rescaled_market(df, scale=10.0):
    """温和缩放：所有价格类列乘以同一系数（保持相对结构和基差比例），
    但绝对量级变了——硬编码的绝对阈值会失效，相对/比例型策略不受影响。"""
    price_cols = ["open","high","low","close","adj_close","spot_close"]
    return df.with_columns([(pl.col(c)*scale).alias(c) for c in price_cols])
