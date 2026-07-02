"""
E3 状态切片回归套件 —— "历史各市况都稳"变成可证伪的门 + 归纳边界钉死。
运行:  pytest tests/test_regime.py -v
"""
import pytest
from quantros.robustness import generate_regime_data, _inject_crash, AlwaysLongStrategy, stress_probe
from quantros.regime import regime_slice_probe, tail_gate, MartingaleDipStrategy
from quantros.overfitting import RobustMomentumStrategy


@pytest.fixture
def crashed():
    """历史里真实出现过一次持续下跌(20天×-3%)——'下行/高波'状态存在。"""
    return _inject_crash(generate_regime_data(drift=0.0012), daily=-0.03, length=20)


@pytest.fixture
def calm():
    return generate_regime_data(drift=0.0012)


def test_momentum_stable_in_all_seen_states(crashed):
    """稳健动量:四个已见状态全稳(崩盘里翻空获利)→ 放行。"""
    blown, _ = regime_slice_probe(RobustMomentumStrategy(5), crashed, verbose=False)
    assert blown is False


def test_martingale_blows_in_down_state(crashed):
    """马丁格尔摊平(散户经典):其余三个状态全部为正、整体曲线好看,
    但'下行/高波'状态深亏 → E3 拦截。这正是'整体好看 ≠ 各市况都稳'的照妖镜。"""
    blown, detail = regime_slice_probe(MartingaleDipStrategy(), crashed, verbose=False)
    assert blown is True and "下行/高波" in detail


def test_always_long_blows_in_down_state(crashed):
    blown, _ = regime_slice_probe(AlwaysLongStrategy(), crashed, verbose=False)
    assert blown is True


def test_e3_BLIND_to_unseen_regimes_e2_covers(calm):
    """归纳边界断言(火鸡问题,记录在案):
    平静历史上满仓多 E3 必然放行(它只能检验出现过的状态)——
    '过去所有状态稳住'≠'未知状态能活';未见 regime 由 E2 压力注入负责拦截。
    若哪天 E3 在平静史上也拦了,说明状态切片机制被改了,需重新审视。"""
    e3_blown, _ = regime_slice_probe(AlwaysLongStrategy(), calm, verbose=False)
    e2_fragile, _ = stress_probe(AlwaysLongStrategy(), calm, verbose=False)
    assert e3_blown is False      # E3 失明:历史里没有它会崩的状态
    assert e2_fragile is True     # E2 拦截:假设的崩盘情景下爆掉


def test_tail_gate_combines_e2_e3(crashed):
    """风险门④组合:E2+E3 任一崩 → tail_ok=False;动量双过 → True。"""
    assert tail_gate(RobustMomentumStrategy(5), crashed) is True
    assert tail_gate(MartingaleDipStrategy(), crashed) is False
