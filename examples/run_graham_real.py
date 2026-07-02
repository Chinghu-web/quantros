"""
格雷厄姆选股 @ 真实沪深300 —— 基本面家族的真实数据落地核对。

运行(凭证只进你自己的终端;START/END 请改成你聚宽账号的实际数据权限窗口):
    cd 项目根目录
    JQ_USER=手机号 JQ_PASS=密码 python3 examples/run_graham_real.py

首次:联网拉 时点成分股+公告日财报(约13个月度快照≈26次调用)+ 全成分价格面板,
      全部缓存 parquet;之后重跑完全离线、免凭证。
产出:五门判决 + data_cache/graham_report.html
"""
import polars as pl

from quantros.fundamentals import fetch_fundamental_store
from quantros.data import from_jq
from quantros.jqcompat import run_jq_grid
from quantros.universal import evaluate_returns
from quantros.verdict import final_verdict
from quantros.htmlreport import write_html_report
from quantros.overfitting import _sharpe

START, END = "2025-03-25", "2026-03-25"        # 账号数据权限窗口:2025-03-24 ~ 2026-03-31
INDEX = "000300.XSHG"

# 格雷厄姆模板,阈值参数化到 g(pb_max / cr_min / stocknum)以解锁 ②③
SRC = """
def initialize(context):
    g.stockindex = '000300.XSHG'
    g.stocknum = 10
    g.pb_max = 2.0
    g.cr_min = 1.2
    g.Transfer_date = (1, 4, 7, 10)
    run_monthly(trade, monthday=20, time='open')

def trade(context):
    if context.current_dt.month not in g.Transfer_date:
        return
    Buylist = check_stocks(context)
    for stock in list(context.portfolio.positions.keys()):
        if stock not in Buylist:
            order_target(stock, 0)
    n = g.stocknum - len(context.portfolio.positions)
    Cash = context.portfolio.cash / n if n > 0 else 0
    for stock in Buylist:
        if len(context.portfolio.positions.keys()) < g.stocknum:
            order_value(stock, Cash)

def check_stocks(context):
    security = get_index_stocks(g.stockindex)
    Stocks = get_fundamentals(query(
            valuation.code, valuation.pb_ratio,
            balance.total_assets, balance.total_liability,
            balance.total_current_assets, balance.total_current_liability
        ).filter(
            valuation.code.in_(security),
            valuation.pb_ratio < g.pb_max,
            balance.total_current_assets/balance.total_current_liability > g.cr_min,
        ))
    Stocks['Debt_Asset'] = Stocks['total_liability']/Stocks['total_assets']
    me = Stocks['Debt_Asset'].median()
    return list(Stocks[Stocks['Debt_Asset'] > me].code)
"""

GRID = [dict(pb_max=p, cr_min=c, stocknum=s)
        for p in (1.5, 2.0) for c in (1.0, 1.2) for s in (5, 10)]


def main():
    print("① 构建时点基本面仓库(首次联网,之后缓存)…")
    store = fetch_fundamental_store(INDEX, START, END)
    codes = sorted(set(store.members["code"].to_list()))
    print(f"   历史全成分并集 {len(codes)} 只")

    print("② 拉全成分价格面板(首次联网,之后缓存)…")
    prices = from_jq(codes, START, END)
    print(f"   {len(prices)} 行 / {prices['symbol'].n_unique()} 只")

    print(f"③ 零改写沙箱重放 {len(GRID)} 组参数(时点成分+公告日财报)…")
    rets = run_jq_grid(SRC, prices, GRID, funda=store, bps=15.0)
    rep = evaluate_returns(rets, sandbox_verified=True)
    v = final_verdict(sandbox_ok=True, pbo=rep["pbo"], dsr=rep["dsr"])

    sh = {n: round(float(_sharpe(r)), 2) for n, r in rets.items()}
    write_html_report("data_cache/graham_report.html",
                      name=f"格雷厄姆选股 @ 真实沪深300({START}~{END})",
                      verdict=v, gate_report=rep,
                      sections=[("各配置净夏普", sh),
                                ("诚实说明", "时点成分股(月度快照)+公告日财报;窗口仅一年,"
                                             "③的样本量提示有效;股票全成本按15bp")])
    print("HTML 报告 → data_cache/graham_report.html")


if __name__ == "__main__":
    main()
