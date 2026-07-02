"""
QuantROS 通用层正门 (Universal Gate) —— 对【任何】策略类型开放的入口。

用户只交"各组合的日收益序列"(纯数字,无代码无 IP),不管背后是期权卖方、
多因子、多腿对冲还是事件循环。在这条(据称的)收益上跑三件事:

  ② PBO (CSCV)          —— 多组合挑冠军,是不是运气?(复用 overfitting.pbo_cscv)
  ③ Deflated Sharpe     —— 冠军的净 edge 扣掉"你试了 N 次"的选择效应后,还显著为正吗?
                            (Bailey & López de Prado 2014;PBO 是相对排名,会放过
                             "稳定亏钱"的组合族——DSR 补这个绝对门)
  bootstrap 显著性       —— 冠军夏普的重采样分布(block,保自相关)

⚠️ 结构性诚实边界(必须显式):
  本层【只能相信你交来的收益】。若回测偷看未来,泄漏同时污染样本内外,
  ②③在假曲线上照样通过——垃圾进垃圾出。因此:
    · 无沙箱背书(sandbox_verified=False)时,判决一律标注"初筛,未验证前视,非实盘结论";
    · 只有收益产自时点化沙箱(quantros.sandbox),①才成立,②③的结论才可信。
  另:必须交【全部试过的组合】(含被淘汰的差组合),只交好的会让 ② 严重低估、
  且 n_trials 低报会让 ③ 虚高——这是用户侧的诚实义务,平台无法代验。
  收益应为【扣除你真实成本后的净收益】,否则 ③ 测的是毛 edge,失真。
"""
import numpy as np
import polars as pl

from quantros.overfitting import pbo_cscv

EULER_GAMMA = 0.5772156649015329


# ── 正态分布工具(无 scipy 依赖)────────────────────────
def _norm_cdf(x):
    from math import erf, sqrt
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _norm_ppf(p):
    """标准正态分位数,Acklam 有理近似(|误差|<1.15e-9),足够判决用。"""
    if not 0.0 < p < 1.0:
        raise ValueError("p 必须在 (0,1)")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow = 0.02425
    if p < plow:
        q = np.sqrt(-2 * np.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > 1 - plow:
        q = np.sqrt(-2 * np.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5; r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def _daily_sharpe(r):
    r = r[~np.isnan(r)]
    if len(r) < 2 or r.std() == 0: return 0.0
    return float(r.mean() / r.std())


# ── ③ Deflated Sharpe Ratio ───────────────────────────
def deflated_sharpe(returns, n_trials, sharpe_variance=None):
    """冠军日收益 → DSR = P[真夏普 > 0 | 试了 n_trials 次的选择效应 + 非正态修正]。

    SR0(噪声下 N 次挑选的期望最大夏普,日频单位):
        SR0 = √V · [ (1−γ)·z(1−1/N) + γ·z(1−1/(N·e)) ]
    DSR = Φ( (SR̂−SR0)·√(T−1) / √(1 − γ₃·SR̂ + (γ₄−1)/4·SR̂²) )
    返回 (dsr, detail_dict)。DSR ≥ 0.95 视为"净 edge 显著"。
    ⚠️ n_trials 必须是你【真正试过】的组合总数——低报=作弊,平台无法代验。
    """
    r = np.asarray(returns, dtype=float); r = r[~np.isnan(r)]
    T = len(r)
    sr = _daily_sharpe(r)
    if T < 30:
        return 0.0, {"sr_daily": sr, "T": T, "note": "样本太短(<30),不足以判定"}
    if n_trials < 1:
        raise ValueError("n_trials 至少为 1")
    if sharpe_variance is None:
        sharpe_variance = 1.0 / T          # 保守:噪声夏普估计量的理论方差 ≈ 1/T
    if n_trials == 1:
        sr0 = 0.0
    else:
        sr0 = float(np.sqrt(sharpe_variance) * ((1 - EULER_GAMMA) * _norm_ppf(1 - 1.0 / n_trials)
                                                + EULER_GAMMA * _norm_ppf(1 - 1.0 / (n_trials * np.e))))
    mu = r.mean(); sd = r.std()
    g3 = float(np.mean(((r - mu) / sd) ** 3))
    g4 = float(np.mean(((r - mu) / sd) ** 4))
    denom = 1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr * sr
    if denom <= 0:                          # 极端偏度/峰度下方差修正失效
        return 0.0, {"sr_daily": sr, "T": T, "note": "收益分布过于极端,DSR 不可靠"}
    z = (sr - sr0) * np.sqrt(T - 1) / np.sqrt(denom)
    dsr = float(_norm_cdf(z))
    return dsr, {"sr_daily": round(sr, 4), "sr_annual": round(sr * np.sqrt(252), 2),
                 "sr0_daily": round(sr0, 4), "T": T, "n_trials": n_trials,
                 "skew": round(g3, 2), "kurt": round(g4, 2)}


# ── bootstrap 冠军夏普(block,保自相关)─────────────────
def bootstrap_sharpe(returns, n_paths=500, block=20, seed=11):
    r = np.asarray(returns, dtype=float); r = r[~np.isnan(r)]
    T = len(r); rng = np.random.default_rng(seed)
    nb = int(np.ceil(T / block))
    out = np.empty(n_paths)
    for i in range(n_paths):
        starts = rng.integers(0, T, size=nb)
        seq = np.concatenate([(s + np.arange(block)) % T for s in starts])[:T]
        out[i] = _daily_sharpe(r[seq]) * np.sqrt(252)
    return {"p5": float(np.percentile(out, 5)), "p50": float(np.percentile(out, 50)),
            "frac_pos": float(np.mean(out > 0))}


# ── 输入装载 ───────────────────────────────────────────
def load_returns_csv(path):
    """长表 CSV → {combo: 日收益数组}。两种列式自动识别:
      combo,date,ret     日收益(首选)
      combo,date,equity  净值曲线(须全程 >0,自动转日收益)
    equity 含 ≤0(如从 0 起步的累计盈亏)→ 明确报错并给出换算指引,不猜。"""
    df = pl.read_csv(path, try_parse_dates=True).sort("date")
    if "ret" in df.columns:
        return {(k[0] if isinstance(k, tuple) else k): g["ret"].to_numpy()
                for k, g in df.group_by("combo", maintain_order=True)}
    if "equity" in df.columns:
        out = {}
        for k, g in df.group_by("combo", maintain_order=True):
            e = g["equity"].to_numpy().astype(float)
            if np.any(e <= 0):
                raise ValueError(
                    "equity 列含 ≤0(疑似从 0 起步的累计盈亏,不是净值)。"
                    "请自行换算:ret = diff(累计盈亏)/初始资金,存成 ret 列——"
                    "资金规模只有你知道,平台不猜")
            out[(k[0] if isinstance(k, tuple) else k)] = e[1:] / e[:-1] - 1.0
        return out
    raise ValueError(f"CSV 需含 ret 或 equity 列,实际列:{df.columns}")


def _to_matrix(returns_by_config):
    names = list(returns_by_config)
    lens = {len(v) for v in returns_by_config.values()}
    if len(lens) != 1:
        raise ValueError(f"各组合收益长度不一致: {sorted(lens)};请先按日期对齐")
    M = np.column_stack([np.asarray(returns_by_config[n], dtype=float) for n in names])
    good = ~np.isnan(M).any(axis=1)
    return M[good], names


# ── 通用层入口 ─────────────────────────────────────────
def evaluate_returns(returns_by_config, *, n_trials=None, sandbox_verified=False,
                     pbo_threshold=0.30, dsr_threshold=0.95, S=10,
                     report_sink=None, verbose=True):
    """通用层正门:{组合名: 日净收益数组} → ②PBO + ③DSR + bootstrap + 分级判决。

    n_trials         : 你【总共】试过的组合数(≥ 提交数;不填按提交数算并提示可能低估)
    sandbox_verified : 收益是否产自时点化沙箱(①);False 时判决强制降级为"初筛"
    """
    M, names = _to_matrix(returns_by_config)
    n_submitted = len(names)
    declared = n_trials if n_trials is not None else n_submitted
    if declared < n_submitted:
        raise ValueError(f"n_trials({declared}) 不能小于提交的组合数({n_submitted})")

    # 单一曲线:② 需要参数族才能评估 → 诚实 BLIND(不报错拒收,也不给假 PBO)
    pbo = None if n_submitted < 2 else pbo_cscv(M, S=S)[0]
    sr_all = np.array([_daily_sharpe(M[:, j]) for j in range(M.shape[1])])
    j = int(np.argmax(sr_all)); champion = names[j]
    dsr, dsr_detail = deflated_sharpe(M[:, j], declared, sharpe_variance=float(np.var(sr_all)) or None)
    boot = bootstrap_sharpe(M[:, j])

    gate2 = None if pbo is None else pbo <= pbo_threshold
    gate3 = dsr >= dsr_threshold
    # 台地识别:②挂但全员为正且③过 → 高 PBO 源于配置近同质(排名即噪声),
    # 语义是"挑冠军没有意义",不是"家族没有 edge"——两者混淆会误导用户
    plateau = bool((gate2 is False) and gate3 and np.all(sr_all > 0))
    trials_note = "" if n_trials is not None else f" ⚠未申报 n_trials,按提交数 {n_submitted} 算(若实际试过更多,③被高估)"
    report = {
        "mode": "universal-gate (returns-only)",
        "sandbox_verified": bool(sandbox_verified),
        "n_submitted": n_submitted, "n_trials": declared,
        "champion": champion,
        "pbo": None if pbo is None else round(float(pbo), 3),
        "gate2_pass": gate2,                              # True/False/None(单曲线=未评估)
        "dsr": round(float(dsr), 3), "gate3_pass": bool(gate3),
        "dsr_detail": dsr_detail, "bootstrap": {k: round(v, 3) for k, v in boot.items()},
        "edge_pass": None if gate2 is None else bool(gate2 and gate3),
        "plateau": plateau,
    }
    if gate2 is None:
        report["single_note"] = ("单一曲线:②过拟合无法评估(需含被淘汰组合的参数族)——"
                                 "未评估 ≠ 通过;建议补交全部试过的组合后重判")
    if plateau:
        report["plateau_note"] = ("②高PBO源于配置近同质(全员为正,排名即噪声):"
                                  "不可宣称冠军配置特殊;若要用,固定任一配置或等权组合,"
                                  "并以该固定选择重新过③——'挑出来的最好'不可信,'随便哪个都行'才可信")
    if gate2 is None:
        report["verdict"] = (f"⚠️ 单一曲线不构成结论:③{'过' if gate3 else '未过'}(DSR={dsr:.3f}),"
                             f"②未评估——补交全部试过的组合才能判" if gate3 else
                             f"❌ 单一曲线且③未过(DSR={dsr:.3f}):现有证据即已不支持实盘")
    elif sandbox_verified:
        report["verdict"] = ("✅ ①②③ 通过:真实、诚实、非过拟合的净 edge(仍需过 ④尾部/⑤容量 风险门才谈实盘)"
                             if report["edge_pass"] else
                             f"❌ 边门未过:{'②PBO' if not gate2 else ''}{' ③DSR' if not gate3 else ''}".strip())
    else:
        report["verdict"] = (("⚠️ 初筛通过(②③),但【未验证前视泄漏】——若回测偷看未来,本结论无效;"
                              "需时点化沙箱背书后才是①②③意义上的通过")
                             if report["edge_pass"] else
                             f"❌ 初筛即未过:{'②PBO ' if not gate2 else ''}{'③DSR' if not gate3 else ''}".strip())

    if verbose:
        tag = "沙箱背书" if sandbox_verified else "无沙箱背书(初筛)"
        print("=" * 64)
        print(f" QuantROS 通用层正门 — {n_submitted} 个组合提交,申报共试 {declared} 次 [{tag}]{trials_note}")
        print("-" * 64)
        if gate2 is None:
            print("  ② PBO        ⚪  单一曲线,无法评估(未评估 ≠ 通过)")
        else:
            print(f"  ② PBO        {'✅' if gate2 else '❌'}  PBO={pbo:.3f} (阈值 {pbo_threshold})")
        print(f"  ③ DSR        {'✅' if gate3 else '❌'}  DSR={dsr:.3f} (阈值 {dsr_threshold}) "
              f"冠军={champion} 年化夏普={dsr_detail.get('sr_annual', '?')}")
        print(f"  bootstrap    冠军夏普 中位={boot['p50']:.2f} 最差5%={boot['p5']:.2f} 为正占比={boot['frac_pos']:.0%}")
        print("-" * 64)
        print(f" 判决:{report['verdict']}")
        if plateau:
            print(f" 注记:{report['plateau_note']}")
        print("=" * 64)
    if report_sink is not None:
        report_sink({k: v for k, v in report.items() if k != "dsr_detail"} | {"dsr_detail": dsr_detail})
    return report


# ── CLI ────────────────────────────────────────────────
def main(argv=None):
    import argparse, json
    p = argparse.ArgumentParser(prog="quantros-gate",
                                description="通用层正门:combo,date,ret 长表 CSV → PBO+DSR 判决(本地跑)")
    p.add_argument("returns_csv", help="列: combo,date,ret(净收益;应含全部试过的组合)")
    p.add_argument("--n-trials", type=int, default=None, help="总共试过的组合数(≥提交数)")
    p.add_argument("--sandbox-verified", action="store_true",
                   help="仅当收益产自时点化沙箱时使用;否则判决为初筛")
    p.add_argument("--out", help="判决 JSON 输出路径(本地)")
    p.add_argument("--html", help="HTML 报告输出路径(本地,可分享)")
    a = p.parse_args(argv)
    rbc = load_returns_csv(a.returns_csv)
    rep = evaluate_returns(rbc, n_trials=a.n_trials, sandbox_verified=a.sandbox_verified)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(rep, f, ensure_ascii=False, indent=2, default=str)
        print(f"判决已写入 {a.out}")
    if a.html:
        from quantros.verdict import final_verdict
        from quantros.htmlreport import write_html_report
        v = final_verdict(sandbox_ok=(True if a.sandbox_verified else None),
                          pbo=rep["pbo"], dsr=rep["dsr"], verbose=False)
        write_html_report(a.html, name=a.returns_csv, verdict=v, gate_report=rep)
        print(f"HTML 报告 → {a.html}")
    return rep


if __name__ == "__main__":
    main()
