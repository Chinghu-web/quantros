"""
QuantROS 期权家族沙箱 (Options Family) —— 第一个深度层家族扩展。

为期权多腿策略(如卖出宽跨式)提供:
  · OptionPITContext —— 时点化期权链:ctx.chain()/price()/otm_leg() 只含 ≤t 的快照,
    前视物理不可能(与主沙箱同一保证,有扰动自证测试);
  · run_option_sandbox —— 逐日重放 {合约: 手数}(负=卖出),按 (t→t+1 价差×乘数)/资金
    记账,含到期内在价值结算与每手交易成本;
  · option_stress_probe —— ④尾部的期权版:对每天的持仓账本,用 BSM 重定价
    "跳空×波动率跳升"情景(从收盘价反解隐波→情景重定价),单日损失超过资金
    阈值即证伪。这是卖权利金策略的头号死法,方向性 E2 替代不了。

⚠️ 诚实边界:
  · 合成链为【平价 BSM + 单一隐波】:无微笑/偏斜、无买卖价差盘口、欧式;
    它用于验证管道与探针逻辑,不用于对真实市场下结论——真实时点化链
    (聚宽/交易所)接入后,同一套代码直接可用(探针只依赖 close 反解隐波);
  · 隐波反解失败(深度虚值価≈0)的合约跳过重定价,计入 detail;
  · 保证金/强平未建模:损失阈值按"占用资金"的比例近似。
"""
import datetime
from math import erf, log, sqrt, exp
import numpy as np
import polars as pl


# ── 自带 BSM(欧式,零依赖)──────────────────────────────
def _ncdf(x): return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def bsm_price(S, K, T, sigma, cp, r=0.02):
    if T <= 0:
        return max(S - K, 0.0) if cp == "C" else max(K - S, 0.0)
    d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    if cp == "C":
        return S * _ncdf(d1) - K * exp(-r * T) * _ncdf(d2)
    return K * exp(-r * T) * _ncdf(-d2) - S * _ncdf(-d1)


def implied_vol(price, S, K, T, cp, r=0.02, lo=0.01, hi=3.0, tol=1e-6):
    """二分反解隐波;价格低于内在价值或无解返回 None(探针会诚实跳过)。"""
    if T <= 0 or price <= 0:
        return None
    if bsm_price(S, K, T, lo, cp, r) > price or bsm_price(S, K, T, hi, cp, r) < price:
        return None
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if bsm_price(S, K, T, mid, cp, r) < price: lo = mid
        else: hi = mid
        if hi - lo < tol: break
    return 0.5 * (lo + hi)


# ── 合成期权市场(验证管道用;真实链接入见诚实边界)──────
def generate_option_market(n_days=500, seed=7, strike_step=100.0, iv_premium=1.2,
                           storm_prob=0.02, storm_len=15, crash=False):
    """MO 量级标的(≈5000 点)+ 逐日全链。波动两状态:平静(0.9%)/风暴(2.5%),
    隐波 = 已实现波动×iv_premium(卖方风险溢价);crash=True 在 70% 处叠加连续下跌。
    返回 (underlying_df, chain_df):
      underlying: trading_date,symbol,close
      chain:      trading_date,code,cp,strike,expiry,close
    """
    rng = np.random.default_rng(seed)
    start = datetime.date(2024, 1, 2)
    dates, spot, vol_state = [], [], []
    S, d, i, storm_left = 5000.0, start, 0, 0
    while len(dates) < n_days:
        d = start + datetime.timedelta(days=i); i += 1
        if d.weekday() >= 5: continue
        if storm_left == 0 and rng.random() < storm_prob:
            storm_left = storm_len
        sig_d = 0.025 if storm_left > 0 else 0.009
        storm_left = max(0, storm_left - 1)
        S *= (1 + rng.normal(0.0004, sig_d))
        dates.append(d); spot.append(S); vol_state.append(sig_d)
    spot = np.array(spot)
    if crash:
        k = int(n_days * 0.7)
        for j in range(k, min(k + 12, n_days)):        # 连续 12 天 -3%:跳空+风暴
            spot[j:] *= 0.97
            vol_state[j] = 0.035
    # 月度到期:每 20 个交易日一个到期日
    expiries = [dates[j] for j in range(19, n_days, 20)]
    und_rows, ch_rows = [], []
    for t in range(n_days):
        d, S = dates[t], float(spot[t])
        und_rows.append({"trading_date": d, "symbol": "MO", "close": S})
        rv = float(np.std(np.diff(np.log(spot[max(0, t - 20):t + 1])))) if t >= 5 else vol_state[t]
        iv = max(0.10, rv * sqrt(252) * iv_premium)
        atm = round(S / strike_step) * strike_step
        alive = [e for e in expiries if e > d][:2]      # 近月+次月
        for e in alive:
            T = (e - d).days / 365.0
            for k_off in range(-6, 7):
                K = atm + k_off * strike_step
                for cp in ("C", "P"):
                    code = f"MO-{cp}-{int(K)}-{e:%Y%m%d}"
                    ch_rows.append({"trading_date": d, "code": code, "cp": cp,
                                    "strike": float(K), "expiry": e,
                                    "close": float(bsm_price(S, K, T, iv, cp))})
    return pl.DataFrame(und_rows), pl.DataFrame(ch_rows)


# ── 时点化期权上下文 ───────────────────────────────────
class OptionPITContext:
    """每个交易日一份【当日】链快照 + ≤t 的标的历史;未来物理不可达。"""
    def __init__(self, underlying, chain):
        self._u_dates = underlying.sort("trading_date")["trading_date"].to_numpy()
        self._u_close = underlying.sort("trading_date")["close"].to_numpy().astype(float)
        self._by_date = {d: g for d, g in chain.group_by("trading_date")}
        self.date = None; self._k = 0; self._snap = None

    def _advance(self, d):
        self.date = d
        self._k = int(np.searchsorted(self._u_dates, d, side="right"))
        key = (d,)
        self._snap = self._by_date.get(key, self._by_date.get(d))

    def spot(self):
        return float(self._u_close[self._k - 1]) if self._k else float("nan")

    def spot_history(self, n):
        return self._u_close[max(0, self._k - int(n)):self._k].copy()

    def chain(self):
        """当日全部挂牌合约快照(polars DataFrame);无则空表。"""
        return self._snap if self._snap is not None else pl.DataFrame()

    def price(self, code):
        if self._snap is None: return float("nan")
        r = self._snap.filter(pl.col("code") == code)
        return float(r["close"][0]) if len(r) else float("nan")

    def otm_leg(self, cp, pos, min_dte=0):
        """虚 pos 档合约(对齐 GetAtmOptionContractByPos(-pos) 语义):
        近月(到期>min_dte 天)里,C 取 ATM+pos 档、P 取 ATM−pos 档。返回 code 或 None。"""
        if self._snap is None or not len(self._snap): return None
        S = self.spot()
        snap = self._snap.filter(
            (pl.col("cp") == cp) & ((pl.col("expiry") - self.date).dt.total_days() > min_dte))
        if not len(snap): return None
        e0 = snap["expiry"].min()
        snap = snap.filter(pl.col("expiry") == e0).sort("strike")
        strikes = snap["strike"].to_numpy()
        atm_i = int(np.argmin(np.abs(strikes - S)))
        j = atm_i + pos if cp == "C" else atm_i - pos
        if j < 0 or j >= len(strikes): return None
        return str(snap["code"][j])

    def dte(self, code):
        if self._snap is None: return None
        r = self._snap.filter(pl.col("code") == code)
        return int((r["expiry"][0] - self.date).days) if len(r) else None


# ── 期权沙箱运行器 ─────────────────────────────────────
def run_option_sandbox(strategy, underlying, chain, *, capital=500_000.0,
                       multiplier=100.0, fee_per_lot=15.0, verbose=True):
    """逐日重放:strategy.on_bar(ctx) 返回目标持仓 {code: 手数}(负=卖出)。
    记账:pnl_t = Σ qty×(price_{t+1}−price_t)×mult − 交易费;到期按内在价值结算。
    返回 {returns, dates, books, net_sharpe, sandbox_verified}。"""
    from quantros.overfitting import _sharpe
    dates = underlying.sort("trading_date")["trading_date"].to_list()
    ctx = OptionPITContext(underlying, chain)
    spec = {r["code"]: (r["cp"], r["strike"], r["expiry"])
            for r in chain.select(["code", "cp", "strike", "expiry"]).unique().iter_rows(named=True)}
    book, books, pnl_by_day = {}, [], []
    px_prev = {}
    for t, d in enumerate(dates):
        ctx._advance(d)
        target = strategy.on_bar(ctx) or {}
        target = {c: float(q) for c, q in target.items() if q}
        # 交易费:按手数变动收
        traded = sum(abs(target.get(c, 0.0) - book.get(c, 0.0)) for c in set(target) | set(book))
        fee = traded * fee_per_lot
        book = target
        books.append((d, dict(book), ctx.spot()))
        # 盯市:t 收盘持仓 → t+1 价差(到期合约按内在价值)
        pnl = -fee
        if t + 1 < len(dates):
            d1 = dates[t + 1]
            px_t = {c: ctx.price(c) for c in book}
            ctx._advance(d1)                       # 仅为取 t+1 价;决策已在 advance 前完成
            S1 = ctx.spot()
            for c, q in book.items():
                p0 = px_t.get(c, float("nan"))
                if p0 != p0: continue
                cp, K, e = spec[c]
                if e <= d1:
                    p1 = max(S1 - K, 0.0) if cp == "C" else max(K - S1, 0.0)   # 到期结算
                else:
                    p1 = ctx.price(c)
                    if p1 != p1: continue
                pnl += q * (p1 - p0) * multiplier
            ctx._advance(d)                        # 回到 t(不影响下轮,防御性)
        pnl_by_day.append(pnl / capital)
    ret = np.array(pnl_by_day[:-1])                # 最后一天无 t+1
    ns = _sharpe(ret)
    if verbose:
        print(f" [期权沙箱] {len(dates)} 日重放 | 净夏普={ns:.2f} 累计={float(np.prod(1+ret)-1):+.1%}")
    return {"returns": ret, "dates": dates[:-1], "books": books,
            "net_sharpe": float(ns), "sandbox_verified": True}


def run_option_sandbox_grid(strategy_cls, underlying, chain, verbose=True, **kw):
    configs = strategy_cls.param_grid() if hasattr(strategy_cls, "param_grid") else [{}]
    out = {}
    for cfg in configs:
        name = ",".join(f"{k}={v}" for k, v in cfg.items()) or "default"
        res = run_option_sandbox(strategy_cls(**cfg), underlying, chain, verbose=False, **kw)
        out[name] = res["returns"]
        if verbose: print(f" [期权沙箱网格] {name}: 净夏普={res['net_sharpe']:.2f}")
    return out


# ── ④尾部·期权版:持仓账本 BSM 重定价 ───────────────────
SCENARIOS = {                                        # (跳空幅度, 隐波倍数)
    "跳空-5%×IV×1.5": (-0.05, 1.5),
    "跳空-8%×IV×2.5": (-0.08, 2.5),                  # 2015/2020 量级
    "跳空+5%×IV×1.5": (+0.05, 1.5),
}


def option_stress_probe(books, chain, *, capital=500_000.0, multiplier=100.0,
                        blowup_frac=0.20, scenarios=SCENARIOS, verbose=True):
    """对每一天的持仓账本:从当日收盘价反解隐波 → 情景(跳空,隐波跳升)重定价,
    单日账本损失 > blowup_frac×资金 → 证伪("一夜爆")。返回 (fragile, detail)。"""
    snap_by_date = {d: g for d, g in chain.group_by("trading_date")}
    worst, worst_info, skipped = 0.0, "", 0
    for d, book, S in books:
        if not book: continue
        key = (d,)
        snap = snap_by_date.get(key, snap_by_date.get(d))
        if snap is None: continue
        rows = {r["code"]: r for r in snap.iter_rows(named=True)}
        for name, (gap, ivx) in scenarios.items():
            loss = 0.0
            for c, q in book.items():
                r = rows.get(c)
                if r is None: continue
                T = (r["expiry"] - d).days / 365.0
                iv = implied_vol(r["close"], S, r["strike"], T, r["cp"])
                if iv is None:
                    skipped += 1; continue
                p2 = bsm_price(S * (1 + gap), r["strike"], T, iv * ivx, r["cp"])
                loss += q * (p2 - r["close"]) * multiplier
            frac = loss / capital
            if frac < worst:
                worst, worst_info = frac, f"{d} {name}"
    fragile = bool(worst < -blowup_frac)
    detail = f"最坏单日={worst:+.1%} @ {worst_info or '无持仓'}(阈值-{blowup_frac:.0%},跳过{skipped}腿)"
    if verbose:
        print(f" [④期权压力] {'⚠️ 一夜爆风险' if fragile else '✓ 情景内扛住'} | {detail}")
    return fragile, detail


# ── 你的策略移植:卖出宽跨式(决策核 1:1,平台 I/O 归沙箱)──
class ShortStrangle:
    """移植自 zhenge_short_strangle.py 的纯逻辑层:
    权利金均线择时 → 双卖虚值 → 止盈35%/止损15% → DTE≤exit_dte 移仓 → 冷却2bar。
    与原版差异(诚实记录):移仓用链上真实 DTE(原 POBO 版以连续无报价代理到期);
    min_credit 用 MO 点数(30);单实例=单组,原'两组并行'由 param_grid 覆盖。"""
    def __init__(self, otm_pos=4, exit_dte=12, profit_target=0.35, stop_loss=0.15,
                 min_credit=30.0, credit_ma=60, cooldown=2, lots=1):
        self.p = dict(otm_pos=otm_pos, exit_dte=exit_dte, profit_target=profit_target,
                      stop_loss=stop_loss, min_credit=min_credit, credit_ma=credit_ma,
                      cooldown=cooldown, lots=lots)
        self.pos = None; self.credit_hist = []; self.cool = 0   # 注意:实例带状态,一次 run 用一个新实例

    @staticmethod
    def param_grid():
        # 原策略真实搜索过的两组 + 邻域(诚实义务:全部空间)
        return [dict(otm_pos=o, exit_dte=e) for o in (2, 3, 4) for e in (9, 12)]

    def on_bar(self, ctx):
        p = self.p
        call = ctx.otm_leg("C", p["otm_pos"], min_dte=p["exit_dte"])
        put = ctx.otm_leg("P", p["otm_pos"], min_dte=p["exit_dte"])
        held = dict(self.pos["book"]) if self.pos else {}
        if call is None or put is None:
            return held
        pc, pp = ctx.price(call), ctx.price(put)
        if pc != pc or pp != pp:
            return held
        credit_now = pc + pp
        self.credit_hist.append(credit_now)
        if len(self.credit_hist) > p["credit_ma"]: self.credit_hist.pop(0)

        if self.pos is not None:                      # 盯市 → 止盈/止损/移仓
            c0, p0 = self.pos["call"], self.pos["put"]
            pcc, ppc = ctx.price(c0), ctx.price(p0)
            dte = ctx.dte(c0)
            if pcc != pcc or ppc != ppc or dte is None:
                self.pos = None; self.cool = p["cooldown"]; return {}
            cur = pcc + ppc; entry = self.pos["entry"]
            if (cur <= entry * (1 - p["profit_target"]) or cur >= entry * (1 + p["stop_loss"])
                    or dte <= p["exit_dte"]):
                self.pos = None; self.cool = p["cooldown"]; return {}
            return held

        if self.cool > 0:                             # 冷却
            self.cool -= 1; return {}
        if len(self.credit_hist) < p["credit_ma"]:    # 均线未热
            return {}
        ma = sum(self.credit_hist) / len(self.credit_hist)
        if credit_now < p["min_credit"] or credit_now <= ma:
            return {}
        n = float(p["lots"])
        book = {call: -n, put: -n}                    # 双卖 n 手
        self.pos = {"call": call, "put": put, "entry": credit_now, "book": book}
        return book
