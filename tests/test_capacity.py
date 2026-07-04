"""
QuantROS 容量探针回归套件 —— 含流动性分离断言 + 模型边界标注。
运行:  pytest tests/test_capacity.py -v
"""
import pytest
import polars as pl
from quantros.capacity import generate_capacity_data, capacity_probe
from quantros.cost import LowTurnoverMomentumStrategy


@pytest.fixture
def liquid():
    return generate_capacity_data(adv_dollars=5e8)


@pytest.fixture
def illiquid():
    return generate_capacity_data(adv_dollars=5e6)


def test_liquid_passes_capacity(liquid):
    """高流动性品种:容量远超目标资金 → 放行。"""
    constrained, cap = capacity_probe(LowTurnoverMomentumStrategy(), liquid, verbose=False)
    assert constrained is False
    assert cap > 1e8


def test_illiquid_caught(illiquid):
    """低流动性品种:同一信号,容量被冲击成本压到目标以下 → 标记规模脆弱。"""
    constrained, cap = capacity_probe(LowTurnoverMomentumStrategy(), illiquid, verbose=False)
    assert constrained is True
    assert cap < 1e8


def test_capacity_scales_with_liquidity(liquid, illiquid):
    """分离断言:容量必须随流动性(ADV)同向放大。
    同一策略、ADV 差 100 倍 → 容量也应差近 100 倍(平方根模型下成正比)。"""
    _, cap_liquid = capacity_probe(LowTurnoverMomentumStrategy(), liquid, verbose=False)
    _, cap_illiquid = capacity_probe(LowTurnoverMomentumStrategy(), illiquid, verbose=False)
    assert cap_liquid > 50 * cap_illiquid


def test_capacity_NA_when_trading_but_no_edge():
    """三态语义:有交易但毛 edge ≤ 0 → 容量 N/A(None),绝不当成'通过'。
    (此前 bug:无 edge 时返回 ∞→PASS,会误导。)"""
    from quantros.robustness import generate_regime_data
    class AlwaysShort:                       # 做空正 drift 市场 → 毛收益必为负
        def generate_signals(self, df):
            return df.with_columns(pl.lit(-1.0).alias("weight"))
    data = generate_regime_data(drift=0.0012)
    constrained, cap = capacity_probe(AlwaysShort(), data, verbose=False)
    assert constrained is None               # N/A,不是 False(通过)
    assert cap != cap                        # nan


def test_capacity_INF_for_no_trading(liquid):
    """诚实边界:不交易的策略没有冲击 → 容量=∞,探针对其失明。
    (容量本就是【模型估计】,非物理真值——这条断言守住退化口径。)"""
    class NeverTradesStrategy:
        def generate_signals(self, dataframe):
            return dataframe.with_columns(pl.lit(0.0).alias("weight"))
    constrained, cap = capacity_probe(NeverTradesStrategy(), liquid, verbose=False)
    assert cap == float("inf")
    assert constrained is False


def test_capacity_bad_adv_not_false_pass():
    """审计修复F8:adv 含 0/NaN 时不能静默假通过/假证伪,应判 N/A(未评估)。"""
    import numpy as np, datetime, polars as pl
    from quantros.capacity import capacity_probe
    d0 = datetime.date(2025, 1, 1); rng = np.random.default_rng(0)
    rows = []
    for i in range(150):
        for s, adv in [("A", 1e9), ("B", 0.0)]:   # B 的 adv=0(坏)
            rows.append({"trading_date": d0 + datetime.timedelta(days=i), "symbol": s,
                         "close": 100 + rng.normal(0, 1), "adv": adv})
    data = pl.DataFrame(rows).with_columns(pl.col("trading_date").cast(pl.Date))
    class AllLong:
        def generate_signals(self, d):
            return d.with_columns(pl.lit(1.0).alias("weight"))
    constrained, cap = capacity_probe(AllLong(), data, verbose=False)
    assert constrained is None or constrained is True   # 绝不能是 False(假通过)
    if constrained is None:
        assert cap != cap                                # N/A → nan
