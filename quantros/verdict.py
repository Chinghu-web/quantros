"""
QuantROS 最终判决 (Final Verdict) —— 证伪器的产品出口。

定位铁律:本平台是【证伪器】,不是保证书。
  · 它能高置信地说"这个策略不能实盘"(被某道门证伪);
  · 它永远不说"这个策略一定赚钱"——最强正向结论只是"未被任何已知测试证伪"。

五道门,两种性质:
  edge 门(回答"有没有真 edge"):
    ① 沙箱(诚实)   收益产自时点化沙箱 → 没偷看未来
    ② PBO (非过拟合) 冠军不是多组合挑出的运气
    ③ DSR (绝对显著) 扣掉搜索次数的选择效应后,净 edge 仍显著为正
      —— ②是相对排名,会放过"稳定亏钱的家族";③补绝对门(有断言为证)
  风险门(回答"这个 edge 在你的规模和坏天气里活不活得下来"):
    ④ 尾部/压力     没见过的崩盘/波动跳升下不爆
    ⑤ 容量          目标资金规模下冲击成本吃不光 edge

判决分级(永不给单一总分):
  ①②③ 有任一 FAIL         → ❌ 已被证伪,不能实盘
  ① 缺失                    → ⚠️ 最高只能到"初筛"——②③在未验证的曲线上不可信
  ①②③ 过、④⑤ 未评估      → ⚠️ 有真 edge,但风险门未评估,不构成实盘依据
  ①②③ 过、④或⑤ FAIL      → ❌ 有 edge 但风险门被证伪(规模/尾部会杀死它)
  五门全过                   → ✅ 未被证伪(可考虑实盘;仍非盈利保证)
"""
PASS, FAIL, UNVERIFIED = "✅ PASS", "❌ FAIL", "⚪ 未评估"


def _tri(v):
    if v is None: return UNVERIFIED
    return PASS if v else FAIL


def final_verdict(*, sandbox_ok, pbo, dsr, pbo_threshold=0.30, dsr_threshold=0.95,
                  tail_ok=None, capacity_ok=None, verbose=True):
    """五门判决。sandbox_ok/tail_ok/capacity_ok: True/False/None(未评估);
    pbo、dsr 传原始数值(便于报告),门内用阈值判。返回判决 dict。"""
    g1 = _tri(sandbox_ok)
    g2 = _tri(None if pbo is None else pbo <= pbo_threshold)
    g3 = _tri(None if dsr is None else dsr >= dsr_threshold)
    g4 = _tri(tail_ok)
    g5 = _tri(capacity_ok)
    gates = {
        "① 沙箱(诚实)": (g1, "收益产自时点化沙箱,前视物理不可能" if g1 == PASS else "收益未经沙箱背书——曲线可能撒谎"),
        "② PBO(非过拟合)": (g2, f"PBO={pbo if pbo is None else round(pbo, 3)} (阈值 {pbo_threshold})"),
        "③ DSR(净edge显著)": (g3, f"DSR={dsr if dsr is None else round(dsr, 3)} (阈值 {dsr_threshold})"),
        "④ 尾部/压力": (g4, "崩盘/波动跳升情景" if g4 != UNVERIFIED else "未评估(未评估 ≠ 通过)"),
        "⑤ 容量": (g5, "目标规模下的冲击成本" if g5 != UNVERIFIED else "未评估(未评估 ≠ 通过)"),
    }

    edge_states = [g1, g2, g3]
    risk_states = [g4, g5]
    if FAIL in edge_states:
        failed = [k for k, (s, _) in gates.items() if s == FAIL and k.startswith(("①", "②", "③"))]
        conclusion = f"❌ 已被证伪({'、'.join(failed)}),不能实盘"
        tier = "falsified"
    elif g1 == UNVERIFIED:
        conclusion = ("⚠️ 初筛(②③基于未验证的曲线):若回测偷看未来,本结论无效;"
                      "需时点化沙箱背书后重判")
        tier = "screening"
    elif UNVERIFIED in edge_states:
        conclusion = "⚠️ edge 门未评齐(缺②或③),不构成结论"
        tier = "incomplete"
    elif FAIL in risk_states:
        failed = [k for k, (s, _) in gates.items() if s == FAIL]
        conclusion = f"❌ 有真 edge,但风险门被证伪({'、'.join(failed)})——该规模/坏天气会杀死它"
        tier = "risk_falsified"
    elif UNVERIFIED in risk_states:
        conclusion = "⚠️ ①②③ 通过:真实、诚实、非过拟合的净 edge;但风险门(④/⑤)未评估,不构成实盘依据"
        tier = "edge_only"
    else:
        conclusion = "✅ 五门全过:未被任何已知测试证伪——可考虑实盘(证伪器不保证盈利)"
        tier = "not_falsified"

    if verbose:
        print("=" * 64)
        print(" QuantROS 最终判决(证伪器:通过 = 没踩到已知的雷,不是保证赚钱)")
        print("-" * 64)
        for name, (state, note) in gates.items():
            print(f"  {name:<14} {state}   {note}")
        print("-" * 64)
        print(f" 结论:{conclusion}")
        print("=" * 64)
    return {"gates": {k: v[0] for k, v in gates.items()}, "tier": tier, "conclusion": conclusion}
