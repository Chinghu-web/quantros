"""
统一入口回归套件 —— 漏斗指南、grid 解析、子命令委托、旗舰 jq 全流程(离线)、契约友好报错。
"""
import datetime
import numpy as np
import polars as pl
import pytest
from quantros.cli import main, parse_grid


def test_no_args_prints_guide(capsys):
    assert main([]) is None
    out = capsys.readouterr().out
    assert "quantros jq" in out and "证伪器" in out and "未评估 ≠ 通过" in out


def test_parse_grid_cartesian():
    g = parse_grid("fast=5,10;slow=20,40,60")
    assert len(g) == 6 and g[0] == dict(fast=5, slow=20)
    assert all(isinstance(c["fast"], int) for c in g)


def test_gate_subcommand_delegates(tmp_path):
    """quantros gate xxx.csv 委托到通用层并返回判决 dict。"""
    rng = np.random.default_rng(0)
    d0 = datetime.date(2025, 1, 1)
    rows = [{"combo": f"c{i}", "date": d0 + datetime.timedelta(days=t),
             "ret": float(rng.normal(0, 0.01))} for i in range(3) for t in range(120)]
    p = tmp_path / "returns.csv"
    pl.DataFrame(rows).write_csv(p)
    rep = main(["gate", str(p)])
    assert "pbo" in rep and "dsr" in rep and rep["sandbox_verified"] is False


def test_jq_flagship_flow_offline(tmp_path, monkeypatch):
    """旗舰:quantros jq 策略.py … 一条命令 → 五门判决 + HTML(from_jq 打桩离线)。"""
    import quantros.data as qd
    from quantros.robustness import generate_regime_data
    monkeypatch.setattr(qd, "from_jq", lambda *a, **k: generate_regime_data(drift=0.0012))
    strat = tmp_path / "s.py"
    strat.write_text("""
def initialize(context):
    g.security = "IF"; g.fast, g.slow = 5, 20
    run_daily(trade)
def trade(context):
    h = attribute_history(g.security, g.slow, "1d", ["close"])
    if len(h["close"]) < g.slow: return
    if h["close"][-g.fast:].mean() > h["close"].mean():
        order_target_value(g.security, context.portfolio.total_value)
    else:
        order_target_value(g.security, 0)
""")
    html = tmp_path / "r.html"
    v = main(["jq", str(strat), "--data", "IF", "--start", "2025-01-01",
              "--end", "2026-08-01", "--grid", "fast=5,10;slow=20,40",
              "--html", str(html)])
    assert v["tier"] in {"falsified", "edge_only", "risk_falsified", "not_falsified"}
    assert html.exists() and "五门明细" in html.read_text(encoding="utf-8")


def test_contract_friendly_error_non_dict():
    """契约校验:on_bar 返回非 dict → 中文报错含类型名和正确示例。"""
    from quantros.sandbox import run_sandbox
    from quantros.robustness import generate_regime_data
    class Bad:
        def on_bar(self, ctx): return [1.0, -1.0]
    with pytest.raises(TypeError, match="必须返回 dict"):
        run_sandbox(Bad(), generate_regime_data(drift=0.0012), verbose=False)


def test_contract_friendly_error_bad_weight():
    from quantros.sandbox import run_sandbox
    from quantros.robustness import generate_regime_data
    class Bad:
        def on_bar(self, ctx): return {s: "多" for s in ctx.symbols}
    with pytest.raises(TypeError, match="必须是数字"):
        run_sandbox(Bad(), generate_regime_data(drift=0.0012), verbose=False)
