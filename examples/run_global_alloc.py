"""
全球资产配置(改进方案)@ 真实复权ETF —— 复权修复后的真实数据落地。
克隆自聚宽 post/75445(作者:牛俊生),固定权重 + 年度再平衡。

运行(在你自己能连东方财富的机器上):
    cd 项目根目录
    python3 examples/run_global_alloc.py

关键:用 from_akshare_etf(默认后复权 hfq)——分红/拆分已还原,不再低估。
(开发机若被代理挡东财源,会取数失败;换能联网的机器即可。)
"""
import numpy as np
import polars as pl

from quantros.data import from_akshare_etf
from quantros.sandbox import run_sandbox, run_sandbox_grid, as_vectorized
from quantros.universal import evaluate_returns
from quantros.regime import tail_gate, regime_slice_probe
from quantros.capacity import capacity_probe
from quantros.verdict import final_verdict
from quantros.overfitting import _sharpe

ETFS = ["510880", "159920", "513100", "511010", "518880", "511880"]  # 红利/港股/纳指/国债/黄金/货基


class GlobalAlloc:
    """固定权重全球配置(改进方案)。逻辑简单,移植零误差。"""
    def __init__(self, dividend=0.25, gold=0.10, bond=0.18):
        self.w = {"510880": dividend, "159920": 0.14, "513100": 0.08,
                  "511010": bond, "518880": gold, "511880": 0.04}

    @staticmethod
    def param_grid():   # 评估"这组权重是不是配置空间里挑出来的运气冠军"
        return [dict(dividend=d, gold=g, bond=b)
                for d in (0.20, 0.25, 0.30) for g in (0.05, 0.10, 0.15) for b in (0.13, 0.18, 0.23)]

    def on_bar(self, ctx):
        return {s: w for s, w in self.w.items() if s in ctx.symbols}


def _perf(r):
    yrs = len(r) / 252; cagr = (1 + (np.prod(1 + r) - 1)) ** (1 / yrs) - 1
    eq = np.cumprod(1 + r); mdd = ((np.maximum.accumulate(eq) - eq) / np.maximum.accumulate(eq)).max()
    return cagr, mdd


def main():
    print("取【后复权】ETF数据(分红已还原)…")
    data = from_akshare_etf(ETFS)                        # 默认 hfq
    print(f"  {len(data)}行 / {data['symbol'].n_unique()}只 / {data['trading_date'].min()}→{data['trading_date'].max()}")

    res = run_sandbox(GlobalAlloc(), data, bps=3.0, verbose=False)
    c, m = _perf(res["returns"])
    print(f"\n默认配置(复权,全样本): 年化={c:.1%} 最大回撤={m:.1%} 净夏普={res['net_sharpe']:.2f}")
    print("(未复权时被低估为年化0.7%/回撤27.5% —— 对比复权修复效果)")

    rets = run_sandbox_grid(GlobalAlloc, data, bps=3.0, verbose=False)
    rep = evaluate_returns(rets, sandbox_verified=True, verbose=False)
    print(f"\n②PBO={rep['pbo']} ③DSR={rep['dsr']} 冠军={rep['champion']}")
    champ = as_vectorized(GlobalAlloc(), data)
    regime_slice_probe(champ, data, bps=3.0)
    t_ok = tail_gate(champ, data, bps=3.0)
    c_con, _ = capacity_probe(champ, data, target_aum=1e7, verbose=False)
    final_verdict(sandbox_ok=True, pbo=rep["pbo"], dsr=rep["dsr"], tail_ok=t_ok,
                  capacity_ok=(None if c_con is None else (c_con is False)))


if __name__ == "__main__":
    main()
