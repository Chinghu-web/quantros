"""品种参数档回归套件 —— 取档、未知名报错、显式 opts 覆盖档值。"""
import pytest
import quantros
from quantros.profiles import get_profile
from quantros.robustness import generate_regime_data
from quantros.overfitting import RobustMomentumStrategy


def test_index_futures_profile_lower_cost():
    """股指期货档的真实成本应显著低于通用档(3bp vs 10bp)。"""
    assert get_profile("cn_index_futures")["realistic_bps"] < get_profile("default")["realistic_bps"]


def test_unknown_profile_raises():
    with pytest.raises(ValueError):
        get_profile("no_such_profile")


def test_explicit_opts_override_profile():
    """显式 opts 优先级高于 profile:同一策略,bps 设极高时成本柱应翻成 FAIL。"""
    data = generate_regime_data(drift=0.0012)
    base = quantros.diagnose(RobustMomentumStrategy, data, profile="cn_index_futures",
                             robustness_paths=40, verbose=False)
    hi = quantros.diagnose(RobustMomentumStrategy, data, profile="cn_index_futures",
                           realistic_bps=99999.0, robustness_paths=40, verbose=False)
    assert base["rows"]["三 · 成本"][0] != hi["rows"]["三 · 成本"][0]
