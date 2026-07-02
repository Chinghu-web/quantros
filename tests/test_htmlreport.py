"""
HTML 报告回归套件 —— 内容完整性 + 免责不可缺席。
"""
from quantros.htmlreport import write_html_report
from quantros.verdict import final_verdict


def test_report_contains_gates_conclusion_disclaimer(tmp_path):
    v = final_verdict(sandbox_ok=True, pbo=0.746, dsr=0.295, tail_ok=True, verbose=False)
    rep = {"n_submitted": 6, "n_trials": 6, "champion": "otm_pos=2,exit_dte=12",
           "pbo": 0.746, "dsr": 0.295, "sandbox_verified": True,
           "dsr_detail": {"sr_annual": -0.18}, "bootstrap": {"p5": -1.49, "p50": -0.12, "frac_pos": 0.45}}
    p = tmp_path / "r.html"
    write_html_report(p, name="测试策略", verdict=v, gate_report=rep,
                      sections=[("各配置净夏普", {"a": -0.18, "b": -0.89})])
    html = p.read_text(encoding="utf-8")
    assert "① 沙箱(诚实)" in html and "② PBO" in html
    assert "已被证伪" in html                       # 结论横幅
    assert "证伪器" in html and "不是保证盈利" in html   # 免责固定出现
    assert "class='fail'" in html and "class='pass'" in html
    assert "0.746" in html and "-0.18" in html


def test_screening_verdict_renders_amber(tmp_path):
    """无沙箱背书 → 初筛(琥珀横幅),报告不得渲染成绿色通过。"""
    v = final_verdict(sandbox_ok=None, pbo=0.05, dsr=0.99, verbose=False)
    p = tmp_path / "s.html"
    write_html_report(p, name="初筛", verdict=v)
    html = p.read_text(encoding="utf-8")
    assert "class='banner b-amber'" in html
    assert "class='banner b-green'" not in html
    assert "初筛" in html
