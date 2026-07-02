"""
期权家族沙箱回归套件 —— PIT 自证(期权版)+ "仓位才是尾部风险"断言 + 全链。
运行:  pytest tests/test_options.py -v
"""
import numpy as np
import polars as pl
import pytest
from quantros.options import (generate_option_market, run_option_sandbox,
                              run_option_sandbox_grid, option_stress_probe,
                              OptionPITContext, ShortStrangle, bsm_price, implied_vol)


@pytest.fixture(scope="module")
def market():
    return generate_option_market(n_days=500)


def test_bsm_and_implied_vol_roundtrip():
    p = bsm_price(5000, 4800, 30 / 365, 0.25, "P")
    iv = implied_vol(p, 5000, 4800, 30 / 365, "P")
    assert abs(iv - 0.25) < 1e-3


def test_option_pit_future_perturbation_invariance(market):
    """沙箱自证(期权版):把后 30% 的链价+标的改成随机数,
    前 70% 的持仓账本必须逐位不变——期权链上下文同样喂不出未来。"""
    und, chain = market
    dates = sorted(und["trading_date"].unique().to_list())
    cut = dates[int(len(dates) * 0.7)]
    rng = np.random.default_rng(5)
    und2 = und.with_columns(pl.when(pl.col("trading_date") > cut)
        .then(pl.col("close") * pl.Series("a", rng.uniform(0.5, 2.0, len(und))))
        .otherwise(pl.col("close")).alias("close"))
    chain2 = chain.with_columns(pl.when(pl.col("trading_date") > cut)
        .then(pl.col("close") * pl.Series("b", rng.uniform(0.5, 2.0, len(chain))))
        .otherwise(pl.col("close")).alias("close"))
    b1 = run_option_sandbox(ShortStrangle(otm_pos=2, exit_dte=9), und, chain, verbose=False)["books"]
    b2 = run_option_sandbox(ShortStrangle(otm_pos=2, exit_dte=9), und2, chain2, verbose=False)["books"]
    for (d1, k1, _), (d2, k2, _) in zip(b1, b2):
        if d1 > cut: break
        assert d1 == d2 and k1 == k2


def test_otm_leg_semantics(market):
    """otm_leg('C', 2) 应取近月、行权价 = ATM+2 档,DTE > min_dte。"""
    und, chain = market
    ctx = OptionPITContext(und, chain)
    d = sorted(und["trading_date"].unique().to_list())[100]
    ctx._advance(d)
    code = ctx.otm_leg("C", 2, min_dte=9)
    assert code is not None
    row = ctx.chain().filter(pl.col("code") == code)
    S = ctx.spot(); atm = round(S / 100.0) * 100.0
    assert float(row["strike"][0]) == atm + 200.0
    assert ctx.dte(code) > 9


def test_strangle_earns_theta_in_calm_loses_edge_in_crash():
    """经济性:平静市吃 theta(夏普>0.8);崩盘市 edge 被打掉(显著低于平静市)。"""
    und_c, ch_c = generate_option_market(n_days=500, crash=False)
    und_x, ch_x = generate_option_market(n_days=500, crash=True)
    calm = run_option_sandbox(ShortStrangle(otm_pos=2, exit_dte=9), und_c, ch_c, verbose=False)
    crash = run_option_sandbox(ShortStrangle(otm_pos=2, exit_dte=9), und_x, ch_x, verbose=False)
    assert calm["net_sharpe"] > 0.8
    assert crash["net_sharpe"] < calm["net_sharpe"] - 0.5


def test_leverage_is_the_tail_risk(market):
    """头条断言:1手 vs 5手,夏普【完全一样】,④重定价一个过一个爆——
    夏普看不见仓位的尾部风险,这正是期权版④存在的理由。"""
    und, chain = market
    r1 = run_option_sandbox(ShortStrangle(otm_pos=2, exit_dte=9, lots=1), und, chain, verbose=False)
    r5 = run_option_sandbox(ShortStrangle(otm_pos=2, exit_dte=9, lots=5), und, chain, verbose=False)
    assert abs(r1["net_sharpe"] - r5["net_sharpe"]) < 0.2          # 夏普分不出二者
    f1, _ = option_stress_probe(r1["books"], chain, verbose=False)
    f5, _ = option_stress_probe(r5["books"], chain, verbose=False)
    assert f1 is False and f5 is True                               # ④分得出


def test_empty_book_not_fragile(market):
    _, chain = market
    fragile, _ = option_stress_probe([], chain, verbose=False)
    assert fragile is False


def test_full_chain_options_to_verdict(market):
    """全链:①期权沙箱网格 → ②③ → 五门。当前合成市上宽跨式家族被②证伪
    (冠军排名不稳)——判决必须落在'已证伪',不含糊。"""
    from quantros.universal import evaluate_returns
    from quantros.verdict import final_verdict
    und, chain = market
    rets = run_option_sandbox_grid(ShortStrangle, und, chain, verbose=False)
    rep = evaluate_returns(rets, sandbox_verified=True, verbose=False)
    res = run_option_sandbox(ShortStrangle(otm_pos=2, exit_dte=9), und, chain, verbose=False)
    t_ok = not option_stress_probe(res["books"], chain, verbose=False)[0]
    v = final_verdict(sandbox_ok=True, pbo=rep["pbo"], dsr=rep["dsr"], tail_ok=t_ok, verbose=False)
    assert v["tier"] == "falsified"
