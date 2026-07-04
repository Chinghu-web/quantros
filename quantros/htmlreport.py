"""
QuantROS HTML 体检报告 —— 把终端判决固化成可保存、可分享的单文件交付物。

设计纪律与判决同源:
  · 永不出现单一总分;逐门 PASS/FAIL/未评估 着色呈现;
  · "未评估 ≠ 通过"与证伪器免责固定出现在报告底部,不可配置删除;
  · 自包含单 HTML(内联 CSS,无外链无 JS),邮件/微信直接发。
"""
import datetime
import html as _html

_CSS = """
body{font-family:-apple-system,'PingFang SC','Microsoft YaHei',sans-serif;
     max-width:860px;margin:24px auto;padding:0 16px;color:#1a1a2e;background:#fafafa}
h1{font-size:22px;border-bottom:3px solid #1a1a2e;padding-bottom:8px}
h2{font-size:16px;margin-top:28px;color:#333}
table{border-collapse:collapse;width:100%;margin:10px 0;background:#fff}
th,td{border:1px solid #ddd;padding:8px 10px;text-align:left;font-size:14px}
th{background:#f0f0f5}
.pass{color:#0a7a2f;font-weight:600}.fail{color:#c0231c;font-weight:600}
.blind{color:#888;font-weight:600}
.banner{padding:14px 18px;border-radius:8px;font-size:16px;font-weight:600;margin:16px 0}
.b-red{background:#fdecea;border:1px solid #c0231c;color:#8a1912}
.b-amber{background:#fff8e1;border:1px solid #b8860b;color:#7a5b00}
.b-green{background:#e8f5e9;border:1px solid #0a7a2f;color:#0a5a23}
.meta{color:#666;font-size:13px}
.frontier{background:#f5f5fa;border-left:4px solid #888;padding:10px 14px;font-size:13px;color:#444}
"""

_DISCLAIMER = ("QuantROS 是【证伪器】,不是保证书:通过 = 未踩到已知的雷,不是保证盈利。"
               "任何'未评估'的门都不等于通过。②③的可信度以①沙箱背书为前提;"
               "n_trials / 参数族完整性 / 数据须【后复权】(未复权会丢分红、系统性低估收益)"
               "均为用户侧诚实义务,平台无法代验。")


def _esc(x):
    return _html.escape(str(x))


def _gate_cls(state):
    if "PASS" in state: return "pass"
    if "FAIL" in state: return "fail"
    return "blind"


def _banner_cls(tier):
    return {"not_falsified": "b-green", "edge_only": "b-amber",
            "screening": "b-amber", "incomplete": "b-amber"}.get(tier, "b-red")


def _kv_table(d):
    rows = "".join(f"<tr><th>{_esc(k)}</th><td>{_esc(v)}</td></tr>" for k, v in d.items())
    return f"<table>{rows}</table>"


def write_html_report(path, *, name, verdict, gate_report=None, sections=None):
    """生成单文件 HTML 报告。
    verdict     : final_verdict() 返回的 dict(gates/tier/conclusion)
    gate_report : 可选,evaluate_returns() 返回的 dict(PBO/DSR/bootstrap 明细)
    sections    : 可选,[(标题, dict 或 str), ...] 附加段(如各配置夏普、归因)
    """
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    parts = [f"<!DOCTYPE html><html lang='zh'><head><meta charset='utf-8'>"
             f"<title>QuantROS 判决 — {_esc(name)}</title><style>{_CSS}</style></head><body>",
             f"<h1>QuantROS 实盘就绪度判决</h1>",
             f"<p class='meta'>策略:{_esc(name)} · 生成:{now} · "
             f"quantros v{__import__('quantros').__version__}</p>",
             f"<div class='banner {_banner_cls(verdict['tier'])}'>{_esc(verdict['conclusion'])}</div>",
             "<h2>五门明细</h2><table><tr><th>门</th><th>判决</th></tr>"]
    for gate, state in verdict["gates"].items():
        parts.append(f"<tr><td>{_esc(gate)}</td>"
                     f"<td class='{_gate_cls(state)}'>{_esc(state)}</td></tr>")
    parts.append("</table>")

    if gate_report:
        parts.append("<h2>通用层明细(②③)</h2>")
        core = {"提交组合数": gate_report.get("n_submitted"),
                "申报总尝试数 n_trials": gate_report.get("n_trials"),
                "冠军": gate_report.get("champion"),
                "② PBO": gate_report.get("pbo"),
                "③ DSR": gate_report.get("dsr"),
                "冠军年化夏普": gate_report.get("dsr_detail", {}).get("sr_annual"),
                "bootstrap(中位/最差5%/为正占比)":
                    " / ".join(str(v) for v in gate_report.get("bootstrap", {}).values()),
                "沙箱背书": "是" if gate_report.get("sandbox_verified") else "否(初筛)"}
        parts.append(_kv_table({k: v for k, v in core.items() if v is not None}))

    for title, body in (sections or []):
        parts.append(f"<h2>{_esc(title)}</h2>")
        parts.append(_kv_table(body) if isinstance(body, dict) else f"<p>{_esc(body)}</p>")

    parts.append(f"<h2>诚实边界</h2><div class='frontier'>{_esc(_DISCLAIMER)}</div>")
    parts.append("</body></html>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(parts))
    return path
