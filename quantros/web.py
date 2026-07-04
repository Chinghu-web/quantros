"""
QuantROS 网页版(通用层 gate)—— 零新依赖,本地一条命令起。

    python -m quantros.web           # 然后浏览器打开 http://127.0.0.1:8000

定位:降门槛的获客入口。用户把"各组合的日收益"粘进来(纯数字,无代码无行情),
出 ②PBO + ③DSR 初筛报告。深度层(沙箱五门/前视物理验证)仍在本地 pip,不上网。

诚实边界(页面上也印着):
  · 本页是【初筛】——未验证前视泄漏,不是实盘结论;
  · 收益须【净】(扣真实成本)、【复权】;组合须含全部试过的(含淘汰的),
    n_trials 报真实搜索次数——这些是平台无法代验的用户诚实义务。
用标准库 http.server,不引入任何 web 框架依赖(保持 quantros 轻量)。
"""
import html
import io
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

_FORM = """<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>QuantROS 证伪器 · 初筛</title><meta name="viewport" content="width=device-width,initial-scale=1">
<style>
body{font-family:-apple-system,'PingFang SC',sans-serif;max-width:760px;margin:24px auto;padding:0 16px;color:#1a1a2e}
h1{font-size:22px}.sub{color:#666;font-size:14px;line-height:1.6}
textarea{width:100%;height:200px;font-family:ui-monospace,Menlo,monospace;font-size:13px;padding:10px;border:1px solid #ccc;border-radius:8px}
label{font-weight:600;font-size:14px}input[type=number]{width:120px;padding:6px;border:1px solid #ccc;border-radius:6px}
button{background:#1a1a2e;color:#fff;border:0;padding:12px 24px;border-radius:8px;font-size:15px;cursor:pointer;margin-top:12px}
.note{background:#fff8e1;border-left:4px solid #b8860b;padding:10px 14px;font-size:13px;color:#7a5b00;border-radius:6px;margin:12px 0}
code{background:#f0f0f5;padding:1px 5px;border-radius:4px}
</style></head><body>
<h1>QuantROS 证伪器 · 初筛</h1>
<p class="sub">回测好看 ≠ 实盘能赚。把你<b>各组参数的日收益</b>贴进来,查两道门:
<b>②是不是运气冠军(PBO)</b>、<b>③扣掉搜索次数后 edge 还显不显著(DSR)</b>。
纯数字、无需代码、不保存。</p>
<div class="note">⚠️ 这是<b>初筛</b>(未验证前视泄漏,非实盘结论)。要可信判决请本地 <code>pip install quantros</code> 走五门沙箱。
收益须<b>净</b>(扣成本)且<b>复权</b>;组合须含<b>全部试过的</b>(含淘汰的差组合)。</div>
<form method="post" action="/diagnose">
<label>粘贴 CSV(表头 <code>combo,date,ret</code>,ret 为日净收益):</label><br>
<textarea name="csv" placeholder="combo,date,ret
a,2024-01-02,0.0012
a,2024-01-03,-0.0008
b,2024-01-02,0.0005
..."></textarea><br>
<label>总共试过多少组参数(n_trials,≥你提交的组合数):</label>
<input type="number" name="n_trials" min="1" placeholder="如 50"><br>
<button type="submit">查两道门 →</button>
</form>
<p class="sub" style="margin-top:24px">开源:github.com/Chinghu-web/quantros · 证伪器只说"不能实盘",通过≠保证赚钱</p>
</body></html>"""


def _parse_form(body):
    """极简 application/x-www-form-urlencoded 解析(只取 csv / n_trials)。"""
    from urllib.parse import parse_qs
    q = parse_qs(body, keep_blank_values=True)
    return q.get("csv", [""])[0], q.get("n_trials", [""])[0]


def _diagnose_html(csv_text, n_trials):
    import polars as pl
    from quantros.universal import evaluate_returns, _to_matrix
    from quantros.verdict import final_verdict
    from quantros.htmlreport import write_html_report
    try:
        df = pl.read_csv(io.StringIO(csv_text), try_parse_dates=True).sort("date")
        rbc = {(k[0] if isinstance(k, tuple) else k): g["ret"].to_numpy()
               for k, g in df.group_by("combo", maintain_order=True)}
        nt = int(n_trials) if str(n_trials).strip() else None
        rep = evaluate_returns(rbc, n_trials=nt, sandbox_verified=False, verbose=False)
        v = final_verdict(sandbox_ok=None, pbo=rep["pbo"], dsr=rep["dsr"], verbose=False)
        import tempfile, os                    # 复用 htmlreport,写临时文件再读回内存
        fd, path = tempfile.mkstemp(suffix=".html")
        os.close(fd)
        write_html_report(path, name="网页初筛", verdict=v, gate_report=rep)
        with open(path, encoding="utf-8") as f:
            body = f.read()
        os.unlink(path)
        return body + '<p style="max-width:760px;margin:16px auto;font-family:sans-serif">' \
                      '<a href="/">← 再测一个</a></p>'
    except Exception as e:
        return (f"<body style='font-family:sans-serif;max-width:700px;margin:40px auto'>"
                f"<h2>解析失败</h2><p>请确认 CSV 表头是 <code>combo,date,ret</code>、日期可解析、ret 是数字。</p>"
                f"<pre>{html.escape(str(e))}</pre><p><a href='/'>← 返回</a></p></body>")


class _Handler(BaseHTTPRequestHandler):
    def _send(self, body, code=200):
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(body.encode("utf-8"))

    def do_GET(self):
        self._send(_FORM)

    def do_POST(self):
        if self.path != "/diagnose":
            self._send(_FORM, 404); return
        n = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n).decode("utf-8")
        csv_text, n_trials = _parse_form(body)
        self._send(_diagnose_html(csv_text, n_trials))

    def log_message(self, *a):
        pass                                 # 静默,不刷屏


def main(host="127.0.0.1", port=8000):
    print(f"QuantROS 网页版(初筛)运行中 → http://{host}:{port}  (Ctrl+C 停止)")
    HTTPServer((host, port), _Handler).serve_forever()


if __name__ == "__main__":
    main()
