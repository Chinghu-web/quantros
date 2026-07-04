"""
QuantROS 中档诊断回归套件 —— 验证"从持仓恢复 PBO"+ 连接接缝只送判决。
运行:  pytest tests/test_multiconfig.py -v
"""
import pytest
import polars as pl
from quantros.multiconfig import diagnose_multiconfig, build_return_matrix
from quantros.overfitting import (
    generate_data_with_edge, RobustMomentumStrategy, OverfitRandomSeedStrategy,
)
from quantros.capacity import generate_capacity_data
from quantros.cost import ChurnyMomentumStrategy, LowTurnoverMomentumStrategy


def _positions(strat, df):
    return strat.generate_signals(df).select(["trading_date", "symbol", "weight"])


def test_pbo_recovered_from_positions_only():
    """核心断言:只给【持仓】、不给代码,也能恢复 PBO 并区分真假。
    真稳健参数族 PBO 低,运气王种子族 PBO 高。"""
    df = generate_data_with_edge(beta=0.3)
    prices = df.select(["trading_date", "symbol", "close"])
    robust = {f"lb{lb}": _positions(RobustMomentumStrategy(lb), df) for lb in (3, 5, 10, 20)}
    lucky = {f"seed{s}": _positions(OverfitRandomSeedStrategy(s), df) for s in range(20)}

    r_robust = diagnose_multiconfig(robust, prices, verbose=False)
    r_lucky = diagnose_multiconfig(lucky, prices, verbose=False)
    assert r_robust["pbo"] < 0.25 and r_robust["overfit_risk"] is False
    assert r_lucky["pbo"] > 0.35 and r_lucky["overfit_risk"] is True


def test_cost_and_capacity_from_positions():
    """成本/容量在中档满血:从持仓即可判定高频空转成本不达标。"""
    df = generate_capacity_data(adv_dollars=5e6, beta=0.05)   # 低流动性 + 弱 edge
    prices = df.select(["trading_date", "symbol", "close", "adv"])
    configs = {"churny": _positions(ChurnyMomentumStrategy(), df),
               "lowturn": _positions(LowTurnoverMomentumStrategy(), df)}
    r = diagnose_multiconfig(configs, prices, verbose=False)
    assert "churny" in r["cost_fail_configs"]          # 高频空转成本脆弱
    assert "churny" in r["capacity_fail_configs"]       # 低流动性下容量受限


def test_sink_receives_only_verdicts_no_ip():
    """连接接缝断言:report_sink 只拿到【判决+指标】,绝不含代码或原始持仓数组。
    这是'本地跑也能连云端、却不泄露 IP'的物理保证。"""
    df = generate_data_with_edge(beta=0.3)
    prices = df.select(["trading_date", "symbol", "close"])
    configs = {f"lb{lb}": _positions(RobustMomentumStrategy(lb), df) for lb in (3, 5, 10, 20)}
    captured = {}
    diagnose_multiconfig(configs, prices, verbose=False, report_sink=lambda rep: captured.update(rep))

    # 送出的每个 per_config 值只能是数字指标(int/float/bool),不能是序列/数组
    for entry in captured["per_config"].values():
        for v in entry.values():
            assert isinstance(v, (int, float, bool))
    assert "pbo" in captured and "一·因果" in captured["not_assessed"]


def test_requires_two_configs():
    """中档(持仓路径)PBO 至少需要 2 个配置;单配置明确报错而非给假结论。
    (收益路径 universal 的单曲线走诚实 BLIND,见 test_universal。)"""
    df = generate_data_with_edge(beta=0.3)
    prices = df.select(["trading_date", "symbol", "close"])
    with pytest.raises(ValueError):
        diagnose_multiconfig({"only": _positions(RobustMomentumStrategy(5), df)}, prices, verbose=False)


def test_sparse_positions_filled_zero_not_dropped():
    """审计修复F1:持仓文件只记非零仓位(缺仓日不写行)是常见写法。
    缺仓日必须填0=空仓,不能留null→NaN→整行被静默丢弃(样本缩短、换手高估)。"""
    df = generate_data_with_edge(beta=0.3)
    prices = df.select(["trading_date", "symbol", "close"])
    full = prices.select(["trading_date", "symbol"]).with_columns(pl.lit(1.0).alias("weight"))
    # c2: symbol 之一只在前半段有持仓行(后半段缺行=空仓)
    sym0 = df["symbol"].unique().to_list()[0]
    mid = sorted(df["trading_date"].unique().to_list())[len(df["trading_date"].unique())//2]
    sparse = full.filter(~((pl.col("symbol") == sym0) & (pl.col("trading_date") > mid)))
    from quantros.multiconfig import build_return_matrix
    M, dates, names = build_return_matrix({"full": full, "sparse": sparse}, prices)
    # 缺仓填0后,两配置的收益矩阵行数应=价格的完整交易日数(未被静默删)
    assert M.shape[0] == df["trading_date"].n_unique() - 1   # 末日无远期收益


def test_duplicate_position_rows_deduped():
    """审计修复F2:持仓表含重复(date,symbol)行,不能让 join 膨胀错位。"""
    df = generate_data_with_edge(beta=0.3)
    prices = df.select(["trading_date", "symbol", "close"])
    pos = prices.select(["trading_date", "symbol"]).with_columns(pl.lit(1.0).alias("weight"))
    pos_dup = pl.concat([pos, pos.head(3)])       # 人为重复前3行
    from quantros.multiconfig import _align_weights
    w = _align_weights(pos_dup, prices)
    assert len(w) == len(prices)                  # 长度不被重复行撑大
