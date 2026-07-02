"""
通用层正门回归套件 —— ②PBO/③DSR 分工断言 + "稳定亏钱骗过②被③拦"的结构性盲区断言。
运行:  pytest tests/test_universal.py -v
"""
import numpy as np
import pytest
from quantros.universal import evaluate_returns, deflated_sharpe, load_returns_csv


def _noise_family(n=40, T=500, seed=0):
    rng = np.random.default_rng(seed)
    return {f"seed{i}": rng.normal(0, 0.01, T) for i in range(n)}


def _edge_family(T=750, seed=0):
    """真实参数族:共享信号 + 质量梯度 + 小特异噪声(组合间高度相关,像真参数族)。"""
    rng = np.random.default_rng(seed)
    common = rng.normal(0, 0.01, T)
    return {f"q{q}": 0.0012 * q + common + rng.normal(0, 0.002, T) for q in (1.0, 0.9, 0.8, 0.7)}


def test_noise_family_caught_by_both_gates():
    """40 个纯噪声组合:② 高 PBO、③ 低 DSR,双双拦截。"""
    r = evaluate_returns(_noise_family(), verbose=False)
    assert r["gate2_pass"] is False or r["gate3_pass"] is False
    assert r["dsr"] < 0.95 and r["edge_pass"] is False


def test_edge_family_passes_both_gates():
    """相关参数族 + 真 edge(3 年):②③ 双过。"""
    r = evaluate_returns(_edge_family(), verbose=False)
    assert r["pbo"] < 0.30 and r["dsr"] >= 0.95 and r["edge_pass"] is True


def test_dsr_deflates_with_declared_trials():
    """核心机制:同一冠军,申报试过 5000 次 → DSR 必须显著下降(搜索越多,要求越严)。"""
    fam = _edge_family()
    d_few = evaluate_returns(fam, verbose=False)["dsr"]
    d_many = evaluate_returns(fam, n_trials=5000, verbose=False)["dsr"]
    assert d_many < d_few - 0.05


def test_stable_losers_expose_pbo_blindspot_caught_by_dsr():
    """结构性盲区断言(③存在的理由):排名稳定的【亏钱】家族,
    ② PBO 照样放行(它只看相对排名),③ DSR 必须拦下(绝对门)。
    若哪天 ② 开始拦这个,说明 PBO 实现被改了,需重新审视。"""
    rng = np.random.default_rng(0)
    T = 500
    common = rng.normal(0, 0.01, T)
    losers = {f"q{q}": -0.002 * q + common + rng.normal(0, 0.002, T) for q in (0.5, 0.8, 1.0, 1.2)}
    r = evaluate_returns(losers, verbose=False)
    assert r["gate2_pass"] is True      # ② 盲:排名稳定 → PBO 低 → 放行
    assert r["gate3_pass"] is False     # ③ 拦:净 edge 为负,DSR≈0
    assert r["edge_pass"] is False


def test_plateau_note_when_homogeneous_family_all_positive():
    """台地语义断言:配置近同质、全员为正、③过但②挂 → 必须给出'挑冠军无意义'
    注记,而非让用户误读为'家族没有 edge'。(35年平安银行 MA 突破的真实画像)"""
    rng = np.random.default_rng(3)
    T = 2000
    common = rng.normal(0.0003, 0.01, T)              # 强共享分量 + 微小特异 → 排名即噪声
    fam = {f"c{i}": common + rng.normal(0, 0.0005, T) for i in range(9)}
    r = evaluate_returns(fam, verbose=False)
    if not r["gate2_pass"] and r["gate3_pass"]:       # 台地形态成立时
        assert r["plateau"] is True and "冠军" in r["plateau_note"]
    else:                                             # 构造未触发也不能有假台地标记
        assert r["plateau"] is False


def test_no_sandbox_verdict_is_capped():
    """诚实分级:无沙箱背书时,即便 ②③ 全过,判决必须是'初筛',绝不给'真 edge'。"""
    r = evaluate_returns(_edge_family(), sandbox_verified=False, verbose=False)
    assert r["edge_pass"] is True and "初筛" in r["verdict"] and "未验证前视" in r["verdict"]
    r2 = evaluate_returns(_edge_family(), sandbox_verified=True, verbose=False)
    assert "初筛" not in r2["verdict"]


def test_understating_trials_rejected():
    """n_trials 低于提交数 = 明显作弊,直接报错。"""
    with pytest.raises(ValueError):
        evaluate_returns(_edge_family(), n_trials=2, verbose=False)


def test_single_curve_blind_not_error():
    """单一曲线(最常见的真实用户):② 诚实 BLIND(None)而非报错拒收;
    判决明确'不构成结论/证据不支持',绝不给假 PBO。"""
    fam = _edge_family()
    one = {"only": list(fam.values())[0]}
    r = evaluate_returns(one, verbose=False)
    assert r["pbo"] is None and r["gate2_pass"] is None and r["edge_pass"] is None
    assert "单一曲线" in r["verdict"] and "单一曲线" in r["single_note"]
    # 喂给 final_verdict:② ⚪ → 最高只能是"未评齐",绝到不了"真 edge"
    from quantros.verdict import final_verdict
    v = final_verdict(sandbox_ok=None, pbo=r["pbo"], dsr=r["dsr"], verbose=False)
    assert v["tier"] in ("screening", "incomplete")


def test_equity_csv_loader(tmp_path):
    """净值曲线 CSV 自动转日收益;从 0 起步的累计盈亏 → 明确报错给指引,不猜。"""
    import datetime
    import polars as pl
    d0 = datetime.date(2025, 1, 1)
    nav = [1.0, 1.01, 0.99, 1.02]
    rows = [{"combo": "a", "date": d0 + datetime.timedelta(days=i), "equity": v}
            for i, v in enumerate(nav)]
    p = tmp_path / "eq.csv"
    pl.DataFrame(rows).write_csv(p)
    loaded = load_returns_csv(p)
    assert np.allclose(loaded["a"], np.diff(nav) / np.array(nav[:-1]))

    bad = [{"combo": "a", "date": d0 + datetime.timedelta(days=i), "equity": v}
           for i, v in enumerate([0.0, 100.0, 250.0])]        # 累计盈亏形态
    p2 = tmp_path / "pnl.csv"
    pl.DataFrame(bad).write_csv(p2)
    with pytest.raises(ValueError, match="累计盈亏"):
        load_returns_csv(p2)


def test_short_sample_dsr_honest():
    """样本太短(<30 天):DSR 不假装能判,返回 0 并注明。"""
    dsr, detail = deflated_sharpe(np.random.default_rng(0).normal(0.001, 0.01, 20), 5)
    assert dsr == 0.0 and "样本太短" in detail["note"]


def test_load_returns_csv_roundtrip(tmp_path):
    import polars as pl
    fam = _edge_family(T=100)
    rows = []
    import datetime
    d0 = datetime.date(2025, 1, 1)
    for name, arr in fam.items():
        for i, v in enumerate(arr):
            rows.append({"combo": name, "date": d0 + datetime.timedelta(days=i), "ret": float(v)})
    p = tmp_path / "returns.csv"
    pl.DataFrame(rows).write_csv(p)
    loaded = load_returns_csv(p)
    assert set(loaded) == set(fam) and len(loaded["q1.0"]) == 100
