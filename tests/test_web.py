"""网页版(通用层gate)回归套件 —— 诊断逻辑 + 免责固定出现 + 坏输入不崩。"""
import datetime
import numpy as np
from quantros.web import _diagnose_html, _parse_form, _FORM


def _csv(configs, T=120, seed=0):
    rng = np.random.default_rng(seed); d0 = datetime.date(2024, 1, 1)
    lines = ["combo,date,ret"]
    for c in configs:
        for i, r in enumerate(rng.normal(0.0002, 0.01, T)):
            lines.append(f"{c},{d0+datetime.timedelta(days=i)},{r:.6f}")
    return "\n".join(lines)


def test_form_has_disclaimer():
    assert "初筛" in _FORM and "未验证前视" in _FORM and "复权" in _FORM


def test_diagnose_returns_report_with_verdict():
    html = _diagnose_html(_csv(["a", "b", "c"]), "30")
    assert "PBO" in html and "DSR" in html
    assert "证伪器" in html and "不是保证盈利" in html      # 免责固定出现
    assert "初筛" in html                                   # 无沙箱背书=初筛


def test_diagnose_bad_csv_no_crash():
    html = _diagnose_html("这不是csv\n乱七八糟", "5")
    assert "解析失败" in html and "combo,date,ret" in html   # 友好报错,不500


def test_parse_form():
    csv, nt = _parse_form("csv=combo%2Cdate%2Cret&n_trials=50")
    assert "combo,date,ret" in csv and nt == "50"
