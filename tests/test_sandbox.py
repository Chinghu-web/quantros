"""
时点化沙箱回归套件 —— 核心是【沙箱自证】:扰动未来,过去决策必须逐位不变。
运行:  pytest tests/test_sandbox.py -v
"""
import numpy as np
import polars as pl
import pytest
from quantros.robustness import generate_regime_data
from quantros.sandbox import run_sandbox, run_sandbox_grid, SandboxMomentum, PITContext, _prepare
from quantros.universal import evaluate_returns


@pytest.fixture
def data():
    return generate_regime_data(drift=0.0012)


def test_sandbox_future_perturbation_cannot_change_past_decisions(data):
    """沙箱自证(与柱子一同一哲学,施加于沙箱自身):
    把后 30% 行情替换成疯狂随机数,前 70% 的每一个决策必须逐位不变。
    这是"前视物理不可能"的可复现实验,不是承诺。"""
    dates = sorted(data["trading_date"].unique().to_list())
    cut = dates[int(len(dates) * 0.7)]
    rng = np.random.default_rng(99)
    noise = pl.Series("n", rng.uniform(0.2, 5.0, len(data)))
    perturbed = data.with_columns(
        pl.when(pl.col("trading_date") > cut)
          .then(pl.col("close") * noise).otherwise(pl.col("close")).alias("close"))
    p1 = run_sandbox(SandboxMomentum(5), data, verbose=False)["positions"]
    p2 = run_sandbox(SandboxMomentum(5), perturbed, verbose=False)["positions"]
    a = p1.filter(pl.col("trading_date") <= cut).sort(["trading_date", "symbol"])
    b = p2.filter(pl.col("trading_date") <= cut).sort(["trading_date", "symbol"])
    assert len(a) > 0 and a.equals(b)


def test_history_physically_bounded(data):
    """ctx.history 无论要多少,只能拿到 ≤ 当前日期的行——上下文里没有未来。"""
    _, per_symbol, all_dates = _prepare(data)
    ctx = PITContext(per_symbol)
    mid = all_dates[len(all_dates) // 2]
    ctx._advance(mid)
    for s in ctx.symbols:
        h = ctx.history(s, 10**9)                      # 贪婪索取
        dates_s, fields_s = per_symbol[s]
        visible = int(np.searchsorted(dates_s, mid, side="right"))
        assert len(h) == visible                       # 恰好 = 截至今日的行数
        assert np.array_equal(h, fields_s["close"][:visible])   # 且内容就是历史前缀


def test_sandbox_accounting_matches_pillars(data):
    """同一动量逻辑:沙箱事件式重放的净夏普应与向量化口径同向且量级一致
    (两者对齐同一 t→t+1 记账;允许小差异来自暖机期处理)。"""
    from quantros.overfitting import RobustMomentumStrategy
    from quantros.cost import _gross_and_turnover
    from quantros.overfitting import _sharpe
    res = run_sandbox(SandboxMomentum(5), data, bps=10.0, verbose=False)
    g, to = _gross_and_turnover(RobustMomentumStrategy(5), data)
    vec_sharpe = _sharpe(g - 10e-4 * to)
    assert res["net_sharpe"] > 1.0
    assert abs(res["net_sharpe"] - vec_sharpe) < 1.0


def test_full_chain_sandbox_to_gates(data):
    """全链:①沙箱网格 → ②PBO ③DSR → edge_pass。真 edge 参数族应三门全过。"""
    rets = run_sandbox_grid(SandboxMomentum, data, bps=10.0, verbose=False)
    rep = evaluate_returns(rets, sandbox_verified=True, verbose=False)
    assert rep["pbo"] < 0.30 and rep["dsr"] >= 0.95 and rep["edge_pass"] is True
    assert "初筛" not in rep["verdict"]


def test_as_vectorized_consistent_with_sandbox(data):
    """as_vectorized 冻结的仓位,经成本柱记账应复现 run_sandbox 的净夏普(同口径)。"""
    from quantros.sandbox import as_vectorized
    from quantros.cost import _gross_and_turnover
    from quantros.overfitting import _sharpe
    res = run_sandbox(SandboxMomentum(5), data, bps=10.0, verbose=False)
    frozen = as_vectorized(SandboxMomentum(5), data)
    g, to = _gross_and_turnover(frozen, data)
    assert abs(_sharpe(g - 10e-4 * to) - res["net_sharpe"]) < 1e-9


def test_ragged_panel_supported(data):
    """参差面板(品种上市日不同,真实数据常态):砍掉一个品种前一半历史,沙箱应正常跑。"""
    sym = data["symbol"].unique().to_list()[0]
    d_mid = sorted(data["trading_date"].unique().to_list())[len(data["trading_date"].unique()) // 2]
    ragged = data.filter(~((pl.col("symbol") == sym) & (pl.col("trading_date") < d_mid)))
    res = run_sandbox(SandboxMomentum(5), ragged, verbose=False)
    assert res["n_days"] > 0 and np.isfinite(res["net_sharpe"])
