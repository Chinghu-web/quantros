"""
最终判决回归套件 —— 五门分级逻辑逐条钉死;核心是"证伪器永不过度承诺"。
运行:  pytest tests/test_verdict.py -v
"""
from quantros.verdict import final_verdict, PASS, FAIL, UNVERIFIED


def test_all_five_pass_is_not_falsified_not_guaranteed():
    v = final_verdict(sandbox_ok=True, pbo=0.05, dsr=0.99, tail_ok=True, capacity_ok=True, verbose=False)
    assert v["tier"] == "not_falsified"
    assert "不保证盈利" in v["conclusion"]      # 最强结论也必须带免责——证伪器铁律


def test_edge_gate_fail_is_falsified():
    v = final_verdict(sandbox_ok=True, pbo=0.60, dsr=0.99, tail_ok=True, capacity_ok=True, verbose=False)
    assert v["tier"] == "falsified" and "②" in v["conclusion"]


def test_dsr_fail_is_falsified_even_with_low_pbo():
    """②过③不过(稳定亏钱家族的画像)→ 证伪。"""
    v = final_verdict(sandbox_ok=True, pbo=0.01, dsr=0.30, verbose=False)
    assert v["tier"] == "falsified" and "③" in v["conclusion"]


def test_no_sandbox_caps_at_screening():
    """①缺失:②③再漂亮也只能是'初筛'——未验证的曲线上不给 edge 结论。"""
    v = final_verdict(sandbox_ok=None, pbo=0.01, dsr=0.99, verbose=False)
    assert v["tier"] == "screening"
    assert v["gates"]["① 沙箱(诚实)"] == UNVERIFIED


def test_edge_pass_risk_unassessed_is_not_live_ready():
    """①②③过、④⑤未评估:明确'不构成实盘依据'——未评估 ≠ 通过。"""
    v = final_verdict(sandbox_ok=True, pbo=0.05, dsr=0.99, verbose=False)
    assert v["tier"] == "edge_only" and "不构成实盘依据" in v["conclusion"]


def test_risk_gate_fail_blocks_despite_true_edge():
    """有真 edge 但尾部门被证伪(卖方策略的典型画像)→ 不能实盘。"""
    v = final_verdict(sandbox_ok=True, pbo=0.05, dsr=0.99, tail_ok=False, capacity_ok=True, verbose=False)
    assert v["tier"] == "risk_falsified" and "④" in v["conclusion"]


def test_gate_states_recorded():
    v = final_verdict(sandbox_ok=True, pbo=0.05, dsr=0.99, tail_ok=True, capacity_ok=None, verbose=False)
    g = v["gates"]
    assert g["① 沙箱(诚实)"] == PASS and g["④ 尾部/压力"] == PASS and g["⑤ 容量"] == UNVERIFIED
    assert v["tier"] == "edge_only"
