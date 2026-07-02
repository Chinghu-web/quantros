"""
QuantROS 状态稳健性探针回归套件 —— 含 E1/E2 分工断言 + block 盲区断言。
运行:  pytest tests/test_robustness.py -v
"""
import pytest
from quantros.robustness import (
    generate_regime_data, bootstrap_robustness_probe, stress_probe, AlwaysLongStrategy,
)
from quantros.overfitting import RobustMomentumStrategy, OverfitRandomSeedStrategy


@pytest.fixture
def df():
    # 正向 drift + 真动量 edge:才能演示"重采样下存活" vs "崩盘里爆"
    return generate_regime_data(drift=0.0012)


def test_robust_passes_both(df):
    """稳健动量:平行历史里稳定为正(E1),压力情景下也不爆(E2)→ 双过。"""
    e1, _ = bootstrap_robustness_probe(RobustMomentumStrategy(5), df, n_paths=100, verbose=False)
    e2, _ = stress_probe(RobustMomentumStrategy(5), df, verbose=False)
    assert e1 is False and e2 is False


def test_no_edge_caught_by_e1(df):
    """无真 edge 的策略:换条历史就坍塌 → E1 拦截。"""
    e1, _ = bootstrap_robustness_probe(OverfitRandomSeedStrategy(0), df, n_paths=100, verbose=False)
    assert e1 is True


def test_E1_E2_division_of_labor(df):
    """分工断言(防巧合绿):永远满仓多在【每一条】平行历史里都稳健(E1放行),
    却死于数据里从没出现过的持续崩盘(E2拦截)。证明 E2 抓到了 E1 漏掉的东西。"""
    e1, _ = bootstrap_robustness_probe(AlwaysLongStrategy(), df, n_paths=100, verbose=False)
    e2, _ = stress_probe(AlwaysLongStrategy(), df, verbose=False)
    assert e1 is False     # E1 失明:满仓多在重采样下看起来很稳
    assert e2 is True      # E2 拦截:崩盘 regime 把它打爆


def test_block_size_is_honest_boundary(df):
    """诚实盲区:block=1 退化成逐点 IID,打碎动量自相关 → 误杀真策略。
    这正是必须用 block(而非 IID)的物理理由。若哪天 block=1 也放行,
    说明重采样实现被改了,需重新审视。"""
    ok_block, _ = bootstrap_robustness_probe(RobustMomentumStrategy(5), df, n_paths=100, block=20, verbose=False)
    iid, _ = bootstrap_robustness_probe(RobustMomentumStrategy(5), df, n_paths=100, block=1, verbose=False)
    assert ok_block is False   # 正确块长:放行真策略
    assert iid is True         # IID:误杀(记录在案的边界)
