"""
QuantROS 实盘就绪度体检报告 (Live-Readiness Report) —— 四柱合一的产品出口。

刻意【不给单一总分】。一个会误导人的"85 分"远比"四张分项判决 + 未评估清单"危险。
每根柱子独立给出三种判决之一:
    ✅ PASS   该维度未发现阻断实盘的硬伤
    ❌ FAIL   该维度存在硬伤,不建议实盘
    ⚪ BLIND  探针对此策略失明(如无参数族无法评估过拟合)——【未评估 ≠ 通过】

总评只回答两件事:有没有硬伤(FAIL) / 有哪些维度没评到(BLIND)。
这与各柱子的'诚实边界'一脉相承:宁可说'我没法判断',也不假装通过。
"""
from quantros.causality import CausalityTester
from quantros.overfitting import oos_degradation_probe, pbo_probe
from quantros.cost import cost_sensitivity_probe
from quantros.capacity import capacity_probe
from quantros.robustness import bootstrap_robustness_probe, stress_probe

PASS, FAIL, BLIND = "✅ PASS", "❌ FAIL", "⚪ BLIND"


def _causality(strategy, df):
    ok = CausalityTester().verify_causality(strategy, df, verbose=False)
    return (PASS if ok else FAIL), ("满足因果律" if ok else "检测到前视泄漏")


def _overfitting(strategy_cls, df):
    if strategy_cls is None or not hasattr(strategy_cls, "param_grid"):
        return BLIND, "无参数族,PBO 无法评估(未评估 ≠ 通过)"
    o1, d1 = oos_degradation_probe(strategy_cls, df, verbose=False)
    o2, pbo = pbo_probe(strategy_cls, df, verbose=False)
    bad = bool(o1) or bool(o2)
    n_cfg = len(strategy_cls.param_grid())
    cfg_note = "" if n_cfg >= 10 else f" ⚠配置少(N={n_cfg}),PBO 偏粗"
    return (FAIL if bad else PASS), f"{d1} | PBO={pbo:.2f}{cfg_note}"


def _cost(strategy, df, realistic_bps):
    frag, detail = cost_sensitivity_probe(strategy, df, realistic_bps=realistic_bps, verbose=False)
    return (FAIL if frag else PASS), detail


def _capacity(strategy, df, target_aum, k):
    if "adv" not in df.columns:
        return BLIND, "数据无 ADV 列,容量无法评估(未评估 ≠ 通过)"
    constrained, cap = capacity_probe(strategy, df, k=k, target_aum=target_aum, verbose=False)
    if constrained is None:
        return BLIND, "有交易但无毛 edge,容量无意义(未评估 ≠ 通过)"
    return (FAIL if constrained else PASS), f"容量≈${cap:.2e} 目标=${target_aum:.0e}"


def _robustness(strategy, df, realistic_bps, n_paths):
    e1, _ = bootstrap_robustness_probe(strategy, df, n_paths=n_paths, bps=realistic_bps, verbose=False)
    e2, _ = stress_probe(strategy, df, bps=realistic_bps, verbose=False)
    bad = bool(e1) or bool(e2)
    return (FAIL if bad else PASS), f"E1 重采样 {'⚠' if e1 else '✓'} / E2 压力 {'⚠' if e2 else '✓'}"


def health_report(strategy_cls, df, *, realistic_bps=10.0,
                  target_aum=1e8, k=0.01, robustness_paths=100, name=None, verbose=True):
    """对单个策略类跑四根柱子,返回 {pillar: (verdict, detail)} 并打印体检卡。

    strategy_cls : 策略【类】。用 strategy_cls() 实例化跑因果/成本/容量;
                   若类带 param_grid 则跑过拟合,否则该柱 BLIND(未评估 ≠ 通过)。
    df           : 数据;含 adv 列才能评估容量。
    """
    strategy = strategy_cls()
    name = name or strategy_cls.__name__
    rows = {
        "一 · 因果性": _causality(strategy, df),
        "二 · 过拟合": _overfitting(strategy_cls, df),
        "三 · 成本":   _cost(strategy, df, realistic_bps),
        "四 · 容量":   _capacity(strategy, df, target_aum, k),
        "五 · 稳健性": _robustness(strategy, df, realistic_bps, robustness_paths),
    }
    fails = [p for p, (v, _) in rows.items() if v == FAIL]
    blinds = [p for p, (v, _) in rows.items() if v == BLIND]
    if fails:
        verdict = f"❌ 存在硬伤,不建议实盘 — 阻断维度:{', '.join(fails)}"
    elif blinds:
        verdict = f"⚠️ 未发现硬伤,但有维度未评估:{', '.join(blinds)}(不等于安全)"
    else:
        verdict = "✅ 四维通过(仍非盈利保证,只是没踩到这四类已知陷阱)"

    if verbose:
        print("=" * 64)
        print(f" QuantROS 实盘就绪度体检报告 — {name}")
        print("-" * 64)
        for pillar, (v, detail) in rows.items():
            print(f"  柱子{pillar:<10} {v}   {detail}")
        print("-" * 64)
        print(f" 总评:{verdict}")
        print("=" * 64)
    return {"rows": rows, "fails": fails, "blinds": blinds, "verdict": verdict}
