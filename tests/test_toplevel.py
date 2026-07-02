"""顶层入口回归套件 —— quantros.diagnose / diagnose_outputs 可用。"""
import quantros
from quantros.robustness import generate_regime_data
from quantros.overfitting import RobustMomentumStrategy, generate_data_with_edge


def test_diagnose_full_mode():
    """完整模式:传策略类 + 数据,返回五柱判决。"""
    data = generate_regime_data(drift=0.0012)
    r = quantros.diagnose(RobustMomentumStrategy, data, robustness_paths=60, verbose=False)
    assert set(["rows", "fails", "blinds", "verdict"]).issubset(r.keys())
    assert "一 · 因果性" in r["rows"]


def test_diagnose_outputs_mode():
    """中档模式:传多配置持仓 + 价格,返回含 PBO 的判决。"""
    df = generate_data_with_edge(beta=0.3)
    prices = df.select(["trading_date", "symbol", "close"])
    pos = {f"lb{lb}": RobustMomentumStrategy(lb).generate_signals(df)
                       .select(["trading_date", "symbol", "weight"]) for lb in (3, 5, 10, 20)}
    r = quantros.diagnose_outputs(pos, prices, verbose=False)
    assert "pbo" in r and r["overfit_risk"] is False
