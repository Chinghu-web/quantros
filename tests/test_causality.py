"""
QuantROS 测谎仪回归套件 —— 含三探针隔离断言 + 已知盲区的诚实标注。
运行：  pip install polars numpy pytest  &&  pytest -v
"""
import pytest
from quantros.causality import CausalityTester, generate_mock_futures_data
from quantros.zoo import (
    HonestBasisStrategy, EvilShiftStrategy, EvilGlobalMeanStrategy,
    EvilPerSymbolRankStrategy, EvilHardcodedConstantStrategy,
)
from quantros.generalization import generalization_probe, make_rescaled_market


@pytest.fixture
def df():
    return generate_mock_futures_data()


# ── 基本正确性 ──────────────────────────────────
def test_honest_passes(df):
    assert CausalityTester().verify_causality(HonestBasisStrategy(), df, verbose=False) is True


def test_shift_caught(df):
    assert CausalityTester().verify_causality(EvilShiftStrategy(), df, verbose=False) is False


def test_global_mean_caught(df):
    assert CausalityTester().verify_causality(EvilGlobalMeanStrategy(), df, verbose=False) is False


def test_per_symbol_rank_caught(df):
    assert CausalityTester().verify_causality(EvilPerSymbolRankStrategy(), df, verbose=False) is False


# ── 探针隔离断言（防巧合绿）────────────────────────
def test_spike_BLIND_to_rank(df):
    """哨兵：只开尖峰探针，对品种内排名必须失明（放行）。"""
    t = CausalityTester(probes={"spike"})
    assert t.verify_causality(EvilPerSymbolRankStrategy(), df, verbose=False) is True


def test_resample_CATCHES_rank(df):
    """哨兵：只开重采样探针，必须独立拦截品种内排名。"""
    t = CausalityTester(probes={"resample"})
    assert t.verify_causality(EvilPerSymbolRankStrategy(), df, verbose=False) is False


# ── 已知盲区：诚实地记录在案 ────────────────────────
def test_dual_probe_is_BLIND_to_hardcoded(df):
    """
    诚实边界断言：双探针对【硬编码未来常数】必然失明（放行）。
    这不是 bug，是动态扰动的理论天花板——常数不走数据通路。
    若哪天这条断言失败（双探针突然抓到了），说明有人改了机制，需重新审视。
    """
    assert CausalityTester().verify_causality(EvilHardcodedConstantStrategy(), df, verbose=False) is True


def test_generalization_probe_flags_hardcoded(df):
    """探针C（启发式）：跨市场泛化测试应把硬编码常数标记为可疑，且不误杀诚实策略。"""
    df2 = make_rescaled_market(df, scale=10.0)
    sus_evil, _ = generalization_probe(EvilHardcodedConstantStrategy(), df, df2, verbose=False)
    sus_honest, _ = generalization_probe(HonestBasisStrategy(), df, df2, verbose=False)
    assert bool(sus_evil) is True
    assert bool(sus_honest) is False
