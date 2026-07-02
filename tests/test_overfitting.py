"""
QuantROS 过拟合探针回归套件 —— 含探针隔离断言 + 诚实盲区标注。
运行:  pytest tests/test_overfitting.py -v
"""
import pytest
from quantros.overfitting import (
    generate_data_with_edge, oos_degradation_probe, pbo_probe,
    RobustMomentumStrategy, OverfitRandomSeedStrategy,
)


@pytest.fixture
def df():
    # 埋有真实动量 edge 的数据(seed 固定 → 结果确定)
    return generate_data_with_edge()


# ── 探针 O1:样本外退化 ─────────────────────────────────
def test_robust_passes_oos(df):
    """诚实动量吃的是真 edge,样本外不崩 → 放行。"""
    overfit, _ = oos_degradation_probe(RobustMomentumStrategy, df, verbose=False)
    assert overfit is False


def test_overfit_caught_oos(df):
    """随机种子伪装的策略样本外坍塌 → 被样本外探针拦截。"""
    overfit, _ = oos_degradation_probe(OverfitRandomSeedStrategy, df, verbose=False)
    assert overfit is True


# ── 探针 O2:回测过拟合概率 PBO ─────────────────────────
def test_robust_low_pbo(df):
    """真 edge → 样本内冠军在样本外仍居前 → PBO 低。"""
    _, pbo = pbo_probe(RobustMomentumStrategy, df, verbose=False)
    assert pbo < 0.25


def test_overfit_high_pbo(df):
    """无真结构 → 样本内冠军样本外掉到中位数附近 → PBO 高(≈0.5)。"""
    _, pbo = pbo_probe(OverfitRandomSeedStrategy, df, verbose=False)
    assert pbo > 0.35


# ── 探针隔离 / 分工断言(防巧合绿)──────────────────────
def test_two_probes_agree_on_overfit(df):
    """哨兵:O1 与 O2 必须对同一只过拟合野兽【独立】给出一致判决。"""
    o1, _ = oos_degradation_probe(OverfitRandomSeedStrategy, df, verbose=False)
    o2, _ = pbo_probe(OverfitRandomSeedStrategy, df, verbose=False)
    assert o1 is True and o2 is True


# ── 诚实盲区:记录在案 ─────────────────────────────────
def test_pbo_BLIND_to_single_config():
    """诚实边界:PBO 需要一个【参数族】才能物理评估。
    对无 param_grid 的单一手调策略必然失明(返回 None),
    与因果性柱子'硬编码常数失明'同理——没有可观测搜索空间就无法判断过拟合。
    若哪天这条断言失败,说明机制被改了,需重新审视。
    """
    class SingleHandTunedStrategy:           # 无 param_grid
        def generate_signals(self, dataframe):
            import polars as pl
            return dataframe.with_columns(pl.lit(1.0).alias("weight"))
    df_edge = generate_data_with_edge()
    overfit, _ = pbo_probe(SingleHandTunedStrategy, df_edge, verbose=False)
    assert overfit is None
