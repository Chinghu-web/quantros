"""
QuantROS 时点化沙箱 (Point-in-Time Sandbox) —— 平台的核:①诚实门。

抓前视泄漏的通用解不是"理解你的策略",而是"控制你能看到的数据":
策略以事件循环方式逐日重放,每个决策时刻只能通过 ctx.history() 拿到
【当日及以前】的数据——未来行根本不在上下文里,前视在物理上不可能发生。
因此它不挑框架:方向、多因子、(接入定价数据后的)期权多腿,都能被 leak-proof。

策略契约(事件式,贴近聚宽/backtrader 用户的心智):
    class MyStrategy:
        def __init__(self, **params): ...
        @staticmethod
        def param_grid(): return [dict(...), ...]     # 可选:解锁 ②③ 全链
        def on_bar(self, ctx) -> dict:                # 每个交易日调用一次
            ctx.date          当前日期
            ctx.symbols       当前有行情的品种列表
            ctx.history(s, n) 品种 s 截至【今日(含)】最近 n 个收盘价(不足则短)
            return {symbol: weight}                   # 当日收盘建仓,吃 t→t+1 收益

沙箱自证(与柱子一同一哲学):对沙箱自身跑"扰动未来→过去决策必须逐位不变"
的物理测试(见 tests/test_sandbox.py)——我们不要求你信任沙箱,我们证明它。

⚠️ 诚实边界:
  · 沙箱保证【时点诚实】,不保证【数据诚实】——若你喂入的行情本身含幸存者
    偏差/错价,沙箱无法察觉(数据审计是另一层,待建);
  · MVP 只支持日频收盘价单字段;期权链/分钟频/多字段是后续家族扩展;
  · 撮合按"收盘价成交 + profile 成本 bp"记账,未建模盘口深度。
"""
import numpy as np
import polars as pl

from quantros.multiconfig import _FrozenStrategy
from quantros.cost import _gross_and_turnover
from quantros.overfitting import _sharpe


class PITContext:
    """时点化数据视图:内部只暴露 ≤ 当前日期的切片,物理上喂不出未来。
    支持多字段(close 必有;open/high/low 等随数据列自动可用)。"""
    def __init__(self, per_symbol):
        self._data = per_symbol            # {sym: (dates_np, {field: np.ndarray})}
        self.date = None
        self.symbols = []                  # 当日有行情的品种
        self._idx = {}                     # {sym: 含当日在内的可见行数}

    def _advance(self, date):
        self.date = date
        self.symbols = []
        for s, (dates, _) in self._data.items():
            k = int(np.searchsorted(dates, date, side="right"))
            self._idx[s] = k
            if k > 0 and dates[k - 1] == date:      # 该品种今天有行情
                self.symbols.append(s)

    def history(self, symbol, n, field="close"):
        """截至今日(含)最近 n 个该字段值;历史不足则返回更短数组;未来物理不可达。"""
        dates, fields = self._data[symbol]
        if field not in fields:
            raise KeyError(f"数据无 '{field}' 列(可用:{sorted(fields)});"
                           f"规范 schema 见 quantros.data")
        k = self._idx.get(symbol, 0)
        return fields[field][max(0, k - int(n)):k].copy()


def _prepare(data):
    data = data.sort(["symbol", "trading_date"])
    fld_cols = [c for c, t in zip(data.columns, data.dtypes)
                if c not in ("trading_date", "symbol", "adv", "volume") and t.is_numeric()]
    per_symbol = {}
    for s in data["symbol"].unique(maintain_order=True).to_list():
        sd = data.filter(pl.col("symbol") == s)
        per_symbol[s] = (sd["trading_date"].to_numpy(),
                         {c: sd[c].to_numpy().astype(float) for c in fld_cols})
    all_dates = np.sort(data["trading_date"].unique().to_numpy())
    return data, per_symbol, all_dates


def run_sandbox(strategy, data, *, bps=10.0, verbose=True):
    """逐日重放策略,返回 {returns, dates, positions, net_sharpe, sandbox_verified}。
    returns 为扣 bps 成本后的组合日净收益(t 日仓位吃 t→t+1 收益,与五柱同口径)。"""
    data, per_symbol, all_dates = _prepare(data)
    ctx = PITContext(per_symbol)
    p_dates, p_syms, p_w = [], [], []
    for d in all_dates:
        ctx._advance(d)
        w = strategy.on_bar(ctx) or {}
        if not isinstance(w, dict):
            raise TypeError(
                f"策略契约:on_bar 必须返回 dict {{symbol: weight}},实际返回 {type(w).__name__}。"
                f"示例:return {{'IF': 1.0, 'IH': -1.0}}(+1满仓多/-1满仓空/0空仓)")
        for s in ctx.symbols:
            try:
                wt = float(w.get(s, 0.0))
            except (TypeError, ValueError):
                raise TypeError(
                    f"策略契约:weight 必须是数字,{d} 品种 {s} 拿到 {w.get(s)!r}。"
                    f"常见原因:返回了 Series/列表而非标量") from None
            p_dates.append(d); p_syms.append(s); p_w.append(wt)
    positions = pl.DataFrame({
        "trading_date": pl.Series(np.array(p_dates)).cast(pl.Date),
        "symbol": p_syms, "weight": p_w,
    })

    # 记账复用成本柱(已被 18 个测试钉死的口径):净 = 毛 − bps×换手
    joined = (data.join(positions, on=["trading_date", "symbol"], how="left")
                  .with_columns(pl.col("weight").fill_null(0.0)))
    frozen = _FrozenStrategy(joined.sort(["symbol", "trading_date"])["weight"].to_numpy())
    g, to = _gross_and_turnover(frozen, data)
    net = g - (bps * 1e-4) * to
    net_sharpe = _sharpe(net)
    if verbose:
        print(f" [沙箱] 重放 {len(all_dates)} 个交易日 × {len(per_symbol)} 品种 | "
              f"净夏普@{bps}bp={net_sharpe:.2f}(时点诚实:✅ 物理保证)")
    return {"returns": net, "positions": positions, "net_sharpe": float(net_sharpe),
            "n_days": int(len(all_dates)), "sandbox_verified": True}


def run_sandbox_grid(strategy_cls, data, *, bps=10.0, verbose=True):
    """整个参数族过沙箱 → {配置名: 日净收益},可直接喂 universal.evaluate_returns
    (sandbox_verified=True),打通 ①沙箱 → ②PBO → ③DSR 全链。"""
    configs = strategy_cls.param_grid() if hasattr(strategy_cls, "param_grid") else [{}]
    out = {}
    for cfg in configs:
        name = ",".join(f"{k}={v}" for k, v in cfg.items()) or "default"
        res = run_sandbox(strategy_cls(**cfg), data, bps=bps, verbose=False)
        out[name] = res["returns"]
        if verbose:
            print(f" [沙箱网格] {name}: 净夏普={res['net_sharpe']:.2f}")
    return out


def as_vectorized(strategy, data):
    """把事件式策略经沙箱重放一次,冻结成向量化接口(_FrozenStrategy),
    从而直接接入 ④(stress/regime)⑤(capacity)等吃 generate_signals 的探针。
    注:冻结的是【该数据上的决策序列】——换数据需重新冻结(压力探针内部会改数据,
    因此对冻结策略,E2 测的是'既定仓位在压力行情下的表现',语义正确)。"""
    res = run_sandbox(strategy, data, verbose=False)
    joined = (data.sort(["symbol", "trading_date"])
                  .join(res["positions"], on=["trading_date", "symbol"], how="left")
                  .with_columns(pl.col("weight").fill_null(0.0)))
    return _FrozenStrategy(joined["weight"].to_numpy())


# ── 示例策略(事件式动量;也是回归测试的样本)────────────
class SandboxMomentum:
    """事件循环写法的动量:只用 ctx.history(≤今日)——聚宽/backtrader 用户的自然写法。"""
    def __init__(self, lookback=5):
        self.lookback = lookback

    @staticmethod
    def param_grid():
        return [dict(lookback=k) for k in (3, 5, 10, 20)]

    def on_bar(self, ctx):
        w = {}
        for s in ctx.symbols:
            c = ctx.history(s, self.lookback + 1)
            if len(c) < self.lookback + 1:
                w[s] = 0.0
                continue
            r = np.diff(c) / c[:-1]
            w[s] = 1.0 if r.mean() > 0 else -1.0
        return w
