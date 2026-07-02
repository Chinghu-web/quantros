"""
QuantROS 成本敏感性探针回归套件 —— 含分离断言 + 退化边界标注。
运行:  pytest tests/test_cost.py -v
"""
import pytest
import polars as pl
from quantros.overfitting import generate_data_with_edge
from quantros.cost import (
    cost_sensitivity_probe, breakeven_bps,
    LowTurnoverMomentumStrategy, ChurnyMomentumStrategy,
)


@pytest.fixture
def df():
    # 用贴近真实的【弱】edge(beta=0.05,毛夏普≈2):此时换手才会致命。
    # 强 edge 会让任何策略都扛得住成本,无法体现探针价值。
    return generate_data_with_edge(beta=0.05)


def test_low_turnover_survives_cost(df):
    """低换手、edge 真:真实成本下净收益仍为正 → 放行。"""
    fragile, _ = cost_sensitivity_probe(LowTurnoverMomentumStrategy(), df, verbose=False)
    assert fragile is False


def test_churny_caught_by_cost(df):
    """毛收益为正但每日翻仓:真实成本把净收益打成负 → 标记成本脆弱。"""
    fragile, _ = cost_sensitivity_probe(ChurnyMomentumStrategy(), df, verbose=False)
    assert fragile is True


def test_breakeven_ordering(df):
    """分离断言:低换手策略的盈亏平衡成本必须显著高于高频空转策略。
    (二者毛夏普相近,差别只在换手——这正是成本探针要照出的东西。)"""
    be_low = breakeven_bps(LowTurnoverMomentumStrategy(), df)
    be_churn = breakeven_bps(ChurnyMomentumStrategy(), df)
    assert be_low > 2 * be_churn


def test_breakeven_INF_for_no_trading(df):
    """诚实边界:从不交易的策略没有换手 → 盈亏平衡成本为 ∞,成本探针对其失明
    (不付成本也不赚钱)。若哪天返回有限值,说明换手口径被改了,需重新审视。"""
    class NeverTradesStrategy:
        def generate_signals(self, dataframe):
            return dataframe.with_columns(pl.lit(0.0).alias("weight"))
    assert breakeven_bps(NeverTradesStrategy(), df) == float("inf")
    fragile, _ = cost_sensitivity_probe(NeverTradesStrategy(), df, verbose=False)
    assert fragile is False
