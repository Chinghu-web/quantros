"""
QuantROS 体检报告回归套件 —— 验证四柱判决正确分流 + 总评诚实。
运行:  pytest tests/test_report.py -v
"""
import pytest
import polars as pl
from quantros.report import health_report, PASS, FAIL, BLIND
from quantros.capacity import generate_capacity_data
from quantros.overfitting import RobustMomentumStrategy
from quantros.cost import ChurnyMomentumStrategy
from quantros.zoo import EvilShiftStrategy


@pytest.fixture
def liquid():
    return generate_capacity_data(adv_dollars=5e8, beta=0.1)


@pytest.fixture
def illiquid():
    return generate_capacity_data(adv_dollars=5e6, beta=0.1)


def test_clean_strategy_all_pass(liquid):
    """稳健动量(带参数族)在高流动性数据上:四维全通过,无硬伤、无未评估。"""
    r = health_report(RobustMomentumStrategy, liquid, verbose=False)
    assert r["fails"] == []
    assert r["blinds"] == []
    assert r["verdict"].startswith("✅")


def test_leak_blocks_despite_pretty_numbers(liquid):
    """前视泄漏策略:即便成本/容量数字漂亮,也必须被因果柱判 FAIL 并阻断。"""
    r = health_report(EvilShiftStrategy, liquid, verbose=False)
    assert r["rows"]["一 · 因果性"][0] == FAIL
    assert "一 · 因果性" in r["fails"]


def test_churny_blocked_on_cost_and_capacity(liquid):
    """高频空转:成本与容量双 FAIL;且无参数族 → 过拟合柱 BLIND(未评估 ≠ 通过)。"""
    r = health_report(ChurnyMomentumStrategy, liquid, verbose=False)
    assert r["rows"]["三 · 成本"][0] == FAIL
    assert r["rows"]["四 · 容量"][0] == FAIL
    assert r["rows"]["二 · 过拟合"][0] == BLIND


def test_capacity_only_blocker_on_illiquid(illiquid):
    """稳健信号跑低流动性:唯一硬伤是容量,总评须明确点名该维度。"""
    r = health_report(RobustMomentumStrategy, illiquid, verbose=False)
    assert "四 · 容量" in r["fails"]
    assert r["rows"]["三 · 成本"][0] == PASS


def test_overfitting_detail_surfaces_confidence(liquid):
    """置信度提示:过拟合判决须带出 PBO 值,且 4 配置时提示'配置少'。"""
    r = health_report(RobustMomentumStrategy, liquid, verbose=False)
    detail = r["rows"]["二 · 过拟合"][1]
    assert "PBO=" in detail and "配置少" in detail
    assert "t=" in detail            # 样本外夏普带 t 值


def test_no_edge_capacity_is_blind():
    """无毛 edge 的策略在报告里容量应为 BLIND,不是 PASS。"""
    import polars as pl
    from quantros.robustness import generate_regime_data
    class AlwaysShort:
        def generate_signals(self, df):
            return df.with_columns(pl.lit(-1.0).alias("weight"))
    r = health_report(AlwaysShort, generate_regime_data(drift=0.0012),
                      robustness_paths=60, verbose=False)
    assert r["rows"]["四 · 容量"][0] == BLIND


def test_blind_is_not_pass(illiquid):
    """诚实总评断言:有 BLIND 但无 FAIL 时,总评必须警示'未评估',绝不报'通过'。"""
    class NoParamButClean:                       # 无 param_grid,信号低换手且因果安全
        def generate_signals(self, df):
            df = df.with_columns(pl.col("close").pct_change().over("symbol").alias("r"))
            df = df.with_columns(pl.col("r").rolling_mean(20).over("symbol").alias("m"))
            return df.with_columns(pl.when(pl.col("m") > 0).then(1.0).otherwise(-1.0).alias("weight"))
    # 用高流动性确保非容量阻断,聚焦验证 BLIND 不被当成 PASS
    r = health_report(NoParamButClean, generate_capacity_data(adv_dollars=5e8, beta=0.1), verbose=False)
    assert BLIND in [v for v, _ in r["rows"].values()]
    if not r["fails"]:
        assert r["verdict"].startswith("⚠️")
