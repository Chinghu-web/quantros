"""
QuantROS × 聚宽真实期权链桥 —— 一条命令:取数(缓存)→ 宽跨式过五门。

用法(凭证只进你自己的终端,不进任何文件/对话):
    cd ~/quantros_project
    JQ_USER=手机号 JQ_PASS=密码 python3 -m quantros.jq_bridge      # 首次:联网拉取并缓存
    python3 -m quantros.jq_bridge                                  # 之后:直接用缓存,离线

数据:复用你已对准字段的 资金费率/data/fetch_jq.py(510300 ETF 期权,
窗口约 2025-03-19 ~ 2026-03-26),转成 quantros options schema 后缓存 parquet。
聚宽查询额度只在首次消耗。

⚠️ 诚实提醒:
  · 这份链是【收盘快照】,无盘中/盘口;成本用 fee_per_lot 近似,ETF 期权
    实际还有约 1~2 tick 价差,判决对成本参数敏感时请自行调高;
  · 窗口仅约 1 年 → ③DSR 的样本量提示会亮;E3 状态切片可能样本不足;
  · 数据来自聚宽,幸存者/复权问题由数据商口径决定,平台不代验。
"""
import os
import sys
from pathlib import Path

import polars as pl

CACHE_DIR = Path(__file__).resolve().parent.parent / "data_cache"
CHAIN_PQ = CACHE_DIR / "jq_510300_chain.parquet"
UND_PQ = CACHE_DIR / "jq_510300_underlying.parquet"
JQ_REPO = Path.home() / "资金费率"

# ETF 量级参数(原策略 510300 版口径)
ETF = dict(multiplier=10000.0, capital=500_000.0, fee_per_lot=3.0, min_credit=0.02)


def fetch_and_cache():
    sys.path.insert(0, str(JQ_REPO))
    from data.fetch_jq import fetch_chain                    # 你已验证过字段的管道
    chain_pd, und_pd, unit = fetch_chain()
    chain = (pl.from_pandas(chain_pd)
             .select([pl.col("date").cast(pl.Date).alias("trading_date"),
                      pl.col("contract_id").alias("code"),
                      pl.col("opt_type").alias("cp"),
                      pl.col("strike").cast(pl.Float64),
                      pl.col("expiry").cast(pl.Date),
                      pl.col("close").cast(pl.Float64)]))
    und = (pl.from_pandas(und_pd)
           .select([pl.col("date").cast(pl.Date).alias("trading_date"),
                    pl.lit("510300").alias("symbol"),
                    pl.col("S").cast(pl.Float64).alias("close")]))
    CACHE_DIR.mkdir(exist_ok=True)
    chain.write_parquet(CHAIN_PQ); und.write_parquet(UND_PQ)
    print(f"已缓存: {len(und)} 个交易日 / {chain['code'].n_unique()} 个合约 / "
          f"{len(chain)} 行链快照 → {CACHE_DIR}(乘数={unit})")
    return und, chain


def load_or_fetch():
    if CHAIN_PQ.exists() and UND_PQ.exists():
        und, chain = pl.read_parquet(UND_PQ), pl.read_parquet(CHAIN_PQ)
        print(f"使用缓存: {len(und)} 交易日 / {chain['code'].n_unique()} 合约(离线)")
        return und, chain
    if not (os.environ.get("JQ_USER") and os.environ.get("JQ_PASS")):
        print("首次运行需要聚宽凭证(只进你自己的终端):\n"
              "  JQ_USER=手机号 JQ_PASS=密码 python3 -m quantros.jq_bridge")
        sys.exit(1)
    return fetch_and_cache()


class ETFStrangle:
    """ShortStrangle 的 ETF 参数化(min_credit=0.02元),网格=原两组+邻域。"""
    @staticmethod
    def param_grid():
        return [dict(otm_pos=o, exit_dte=e) for o in (2, 3, 4) for e in (9, 12)]

    def __new__(cls, otm_pos=2, exit_dte=9):
        from quantros.options import ShortStrangle
        return ShortStrangle(otm_pos=otm_pos, exit_dte=exit_dte,
                             min_credit=ETF["min_credit"])


def main(argv=None):
    from quantros.options import run_option_sandbox, run_option_sandbox_grid, option_stress_probe
    from quantros.universal import evaluate_returns
    from quantros.verdict import final_verdict

    und, chain = load_or_fetch()
    kw = dict(capital=ETF["capital"], multiplier=ETF["multiplier"], fee_per_lot=ETF["fee_per_lot"])

    print("\n① 时点化沙箱重放参数族(真实 510300 期权链)…")
    rets = run_option_sandbox_grid(ETFStrangle, und, chain, **kw)
    rep = evaluate_returns(rets, sandbox_verified=True)

    champ_cfg = dict(zip(("otm_pos", "exit_dte"),
                         (int(rep["champion"].split(",")[0].split("=")[1]),
                          int(rep["champion"].split(",")[1].split("=")[1]))))
    res = run_option_sandbox(ETFStrangle(**champ_cfg), und, chain, verbose=False, **kw)
    frag, det = option_stress_probe(res["books"], chain,
                                    capital=ETF["capital"], multiplier=ETF["multiplier"])
    v = final_verdict(sandbox_ok=True, pbo=rep["pbo"], dsr=rep["dsr"], tail_ok=(not frag))

    from quantros.htmlreport import write_html_report
    from quantros.overfitting import _sharpe
    out = CACHE_DIR / "quantros_report.html"
    write_html_report(out, name="卖出宽跨式 @ 真实510300期权链", verdict=v, gate_report=rep,
                      sections=[("各配置净夏普(沙箱重放)",
                                 {n: round(float(_sharpe(r)), 2) for n, r in rets.items()}),
                                ("④期权压力明细", det)])
    print(f"\nHTML 报告 → {out}")
    return v


if __name__ == "__main__":
    main()
