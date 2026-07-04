"""
全球资产配置 @ 真实复权ETF(聚宽源)—— 绕开东财,用聚宽后复权数据。
克隆自聚宽 post/75445。聚宽服务器在多数网络下可达(不走东方财富)。

运行(凭证只进你自己终端;窗口须在你账号数据权限内):
    cd ~/quantros_project
    JQ_USER=手机号 JQ_PASS=密码 python3 examples/run_global_alloc_jq.py

⚠️ 账号一年窗口 → 样本短,③DSR 会亮"样本不足";本次主要验证:
   复权后收益率是否正常(不再被除息跳空低估成年化0.7%)。
"""
import numpy as np
import polars as pl

from quantros.data import from_jq
from quantros.sandbox import run_sandbox, run_sandbox_grid, as_vectorized
from quantros.universal import evaluate_returns
from quantros.regime import tail_gate
from quantros.capacity import capacity_probe
from quantros.verdict import final_verdict
from quantros.overfitting import _sharpe

START, END = "2025-03-25", "2026-03-25"        # 账号数据权限窗口内(2025-03-24~2026-03-31)
ETFS = ["510880.XSHG", "159920.XSHE", "513100.XSHG", "511010.XSHG", "518880.XSHG", "511880.XSHG"]


class GlobalAlloc:
    def __init__(self, dividend=0.25, gold=0.10, bond=0.18):
        self.w = {"510880.XSHG": dividend, "159920.XSHE": 0.14, "513100.XSHG": 0.08,
                  "511010.XSHG": bond, "518880.XSHG": gold, "511880.XSHG": 0.04}

    @staticmethod
    def param_grid():
        return [dict(dividend=d, gold=g, bond=b)
                for d in (0.20, 0.25, 0.30) for g in (0.05, 0.10, 0.15) for b in (0.13, 0.18, 0.23)]

    def on_bar(self, ctx):
        return {s: w for s, w in self.w.items() if s in ctx.symbols}


def main():
    print(f"聚宽后复权取数(分红已还原)… {START} → {END}")
    data = from_jq(ETFS, START, END)            # from_jq 已默认 fq=post 后复权
    print(f"  {len(data)}行 / {data['symbol'].n_unique()}只ETF")

    res = run_sandbox(GlobalAlloc(), data, bps=3.0, verbose=False)
    r = res["returns"]; yrs = len(r) / 252
    cagr = (1 + (np.prod(1 + r) - 1)) ** (1 / max(yrs, 0.1)) - 1
    eq = np.cumprod(1 + r); mdd = ((np.maximum.accumulate(eq) - eq) / np.maximum.accumulate(eq)).max()
    print(f"\n默认配置(复权,{len(r)}个交易日): 年化≈{cagr:.1%} 最大回撤={mdd:.1%} 净夏普={res['net_sharpe']:.2f}")
    print("(对照:未复权曾被低估为年化0.7% —— 看复权后是否正常)")

    rets = run_sandbox_grid(GlobalAlloc, data, bps=3.0, verbose=False)
    rep = evaluate_returns(rets, sandbox_verified=True, verbose=False)
    print(f"\n②PBO={rep['pbo']} ③DSR={rep['dsr']}(一年样本,DSR样本不足属正常)")
    champ = as_vectorized(GlobalAlloc(), data)
    t_ok = tail_gate(champ, data, bps=3.0)
    c_con, _ = capacity_probe(champ, data, target_aum=1e7, verbose=False)
    final_verdict(sandbox_ok=True, pbo=rep["pbo"], dsr=rep["dsr"], tail_ok=t_ok,
                  capacity_ok=(None if c_con is None else (c_con is False)))


if __name__ == "__main__":
    main()
