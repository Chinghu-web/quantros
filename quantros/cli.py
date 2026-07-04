"""
QuantROS 统一入口 —— 一个 `quantros` 命令收拢全部路径。

    quantros                       看漏斗指南(你有什么 → 走哪条路)
    quantros jq 策略.py --data 510300.XSHG --start 2023-01-01 --end 2025-12-31 \\
             --grid "fast=5,10;slow=20,40,60"
                                   ★ 旗舰:聚宽用户一条命令 = 取数(缓存)→零改写沙箱
                                     →②③→五门判决→HTML 报告
    quantros gate returns.csv      收益序列 → ②③ 初筛(任何策略类型)
    quantros positions pos.csv px.csv   多配置持仓 → PBO+成本+容量
    quantros check                 真实数据端到端自检
"""
import re
import sys
from itertools import product

GUIDE = """\
QuantROS 证伪器 —— 你有什么,就走哪条路:

  ① 有聚宽策略文件 + 聚宽账号(推荐,可信判决):
     quantros jq 策略.py --data 510300.XSHG --start 2023-01-01 --end 2025-12-31 \\
              --grid "fast=5,10;slow=20,40,60"
     (--grid 必须是你【真正搜索过的全部参数组合】,少报=③被高估)

  ② 只有回测结果曲线(任何框架,10 分钟,初筛):
     quantros gate returns.csv --n-trials 500 --html report.html
     (列:combo,date,ret 或 combo,date,equity;净值/净收益;含被淘汰的差组合;
      单条曲线也收——②会诚实标注"无法评估";导出教程见 EXPORT.md)

  ③ 只有持仓序列:
     quantros positions positions.csv prices.csv

  ④ 网页版(降门槛,浏览器贴收益出初筛报告):
     quantros web      # 然后打开 http://127.0.0.1:8000

  ⑤ 环境自检(需联网):
     quantros check

  Python API / 手工迁移 / 期权家族:见 README.md 与 MIGRATION.md
  纪律:证伪器只说"不能实盘",通过 ≠ 保证盈利;未评估 ≠ 通过。"""


def _num(x):
    try:
        return int(x)
    except ValueError:
        try:
            return float(x)
        except ValueError:
            return x


def parse_grid(s):
    """'fast=5,10;slow=20,40,60' → 笛卡尔积 [{'fast':5,'slow':20}, ...]"""
    parts = [p for p in re.split(r"[;\s]+", s.strip()) if p]
    keys, vals = [], []
    for part in parts:
        k, v = part.split("=", 1)
        keys.append(k); vals.append([_num(x) for x in v.split(",")])
    return [dict(zip(keys, combo)) for combo in product(*vals)]


def _jq_flow(rest):
    import argparse
    p = argparse.ArgumentParser(prog="quantros jq",
                                description="聚宽策略一条命令过五门(零改写)")
    p.add_argument("strategy", help="聚宽策略 .py 文件(原样,不用改)")
    p.add_argument("--data", required=True, help="标的,逗号分隔(聚宽代码,如 510300.XSHG)")
    p.add_argument("--start", required=True); p.add_argument("--end", required=True)
    p.add_argument("--grid", required=True,
                   help='全部搜索过的参数,如 "fast=5,10;slow=20,40,60"(覆盖到 g)')
    p.add_argument("--n-trials", type=int, default=None,
                   help="总尝试次数(≥grid 数;含 grid 之外试过又放弃的)")
    p.add_argument("--bps", type=float, default=10.0)
    p.add_argument("--capital", type=float, default=1e6)
    p.add_argument("--html", default="quantros_report.html")
    a = p.parse_args(rest)

    import quantros.data as qd
    from quantros.jqcompat import run_jq_grid
    from quantros.universal import evaluate_returns
    from quantros.verdict import final_verdict
    from quantros.htmlreport import write_html_report

    grid = parse_grid(a.grid)
    print(f"取数(首次联网,之后缓存离线)… {a.data} {a.start}→{a.end}")
    data = qd.from_jq(a.data.split(","), a.start, a.end)
    print(f"① 零改写沙箱重放 {len(grid)} 组参数…")
    rets = run_jq_grid(a.strategy, data, grid, capital=a.capital, bps=a.bps)
    rep = evaluate_returns(rets, n_trials=a.n_trials, sandbox_verified=True)
    v = final_verdict(sandbox_ok=True, pbo=rep["pbo"], dsr=rep["dsr"])
    from quantros.overfitting import _sharpe
    write_html_report(a.html, name=a.strategy, verdict=v, gate_report=rep,
                      sections=[("各配置净夏普(沙箱重放)",
                                 {n: round(float(_sharpe(r)), 2) for n, r in rets.items()})])
    print(f"HTML 报告 → {a.html}")
    return v


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "guide"):
        print(GUIDE); return None
    cmd, rest = argv[0], argv[1:]
    if cmd == "jq":
        return _jq_flow(rest)
    if cmd == "gate":
        from quantros.universal import main as m; return m(rest)
    if cmd == "positions":
        from quantros.multiconfig import main as m; return m(rest)
    if cmd == "check":
        from quantros.check import main as m; return m(rest)
    if cmd == "web":
        from quantros.web import main as m
        return m(port=int(rest[0]) if rest else 8000)
    print(f"未知子命令:{cmd}\n"); print(GUIDE); sys.exit(2)


if __name__ == "__main__":
    main()
