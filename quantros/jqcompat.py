"""
QuantROS 聚宽兼容垫片 (JQ Compat Shim) —— 零改写跑聚宽策略。

思路:不翻译用户代码,而是【翻译运行时】——实现聚宽同名 API(g / attribute_history /
order_target_value / run_daily / handle_data / log / set_*),底层接时点化沙箱。
用户的策略文件原封不动,verbatim 执行;垫片自身就是一个沙箱策略(JQAdapter.on_bar),
因此 ①前视物理不可能 的保证自动继承,扰动自证测试同样适用。

支持子集(日频,覆盖绝大多数聚宽日线策略):
  initialize(context) / handle_data(context, data) / run_daily(func, ...)
  g 全局命名空间 / log.info|warn|error
  attribute_history(sec, n, '1d', ['close'])  ← 聚宽语义:【不含当日】,垫片忠实保留
  history(n, '1d', 'close', [secs])
  get_current_data()[sec].last_price          ← 当日收盘(与"收盘决策"口径一致)
  order_target_value / order_target / order_value / order(按股数)
  set_benchmark / set_option / set_order_cost / set_slippage / set_universe(记录性 no-op)

⚠️ 诚实边界:
  · 不支持的 API(get_price / get_fundamentals / 分钟频 / 融资融券等)一律
    UnsupportedJQAPI 大声报错并列名——绝不静默假装支持;
  · portfolio.total_value 固定为初始资金(不复利滚动仓位),权重口径与五柱一致;
  · 决策用 ≤昨日 数据、当日收盘成交(比聚宽某些撮合多一天滞后,偏保守方向);
  · 数据 symbol 需用聚宽代码(如 '510300.XSHG'),与用户策略里的写法一致。
"""
import datetime as _datetime

import numpy as np

from quantros.sandbox import run_sandbox


def datetime_parse(x):
    if isinstance(x, _datetime.datetime):
        return x.date()
    if isinstance(x, _datetime.date):
        return x
    return _datetime.date.fromisoformat(str(x)[:10])


class UnsupportedJQAPI(RuntimeError):
    pass


class _G:                                       # 聚宽的 g 全局对象
    pass


class _Log:
    def info(self, *a, **k): pass
    warn = error = debug = info


class _Position:
    def __init__(self, adapter, sec):
        w = adapter._weights.get(sec, 0.0)
        px = adapter._px_now(sec)
        self.value = w * adapter._capital
        self.total_amount = self.closeable_amount = (
            self.value / px if px == px and px > 0 else 0.0)
        self.price = px


class _PositionsView:
    """聚宽 positions 语义:未持有的标的返回零仓位对象(而非 KeyError)。"""
    def __init__(self, adapter): self._a = adapter
    def __getitem__(self, sec): return _Position(self._a, sec)
    def __contains__(self, sec): return self._a._weights.get(sec, 0.0) != 0.0
    def keys(self): return [s for s, w in self._a._weights.items() if w != 0.0]
    def __len__(self): return len(self.keys())
    def __iter__(self): return iter(self.keys())
    def items(self): return [(s, _Position(self._a, s)) for s in self.keys()]
    def values(self): return [_Position(self._a, s) for s in self.keys()]


class _Portfolio:
    """活账户:cash/市值随下单变化(常量资金口径,不复利)。
    否则 order_value(sec, cash) 全仓策略会反复叠加成隐性杠杆,判决失真。"""
    def __init__(self, adapter):
        self._a = adapter
        self.positions = _PositionsView(adapter)

    @property
    def total_value(self): return self._a._capital

    @property
    def positions_value(self):
        return self._a._capital * sum(w for w in self._a._weights.values() if w > 0)

    @property
    def available_cash(self):
        return max(0.0, self._a._capital - self.positions_value)

    cash = available_cash                    # 聚宽旧别名 context.portfolio.cash


class _Context:
    def __init__(self, adapter):
        self.portfolio = _Portfolio(adapter)
        self.current_dt = None
        self.previous_date = None


class _Bar:
    def __init__(self, price, day_open=None):
        self.close = price
        self.last_price = price
        self.price = price
        self.day_open = day_open if day_open is not None else price
        self.paused = False        # 停牌日不在当日行情里 → ctx.symbols 已天然剔除


def _unsupported(name, msg=None):
    def _raise(*a, **k):
        raise UnsupportedJQAPI(msg or (
            f"聚宽 API `{name}` 垫片暂不支持(诚实拒绝,不静默假装)。"
            f"支持子集见 quantros/jqcompat.py 文档;日线决策请用 attribute_history。"))
    return _raise


_FUNDAMENTAL_MSG = (
    "`{name}` 需要【时点化】数据才能诚实评估:成分股按当日名单(否则=幸存者偏差,"
    "用今天的沪深300名单跑历史,死掉/调出的股票被系统性剔除,收益必然虚高)、"
    "财报按【公告日】可见(否则=前视,财报期末≠市场知道的日子)。"
    "垫片没有这两样时点数据,跑出来的判决必然虚高——宁拒绝,不出假判决。"
    "基本面家族在路线图上;当下可走:聚宽研究环境导出多组参数回测收益 → quantros gate(初筛)。")

_UNSUPPORTED = {
    "get_price": None, "get_ticks": None, "get_trade_days": None,
    "get_industry_stocks": None, "margincash_open": None,
    "order_percent": None, "order_target_percent": None, "inout_cash": None,
    "get_all_securities": None, "get_security_info": None,
    "get_fundamentals": _FUNDAMENTAL_MSG.format(name="get_fundamentals"),
    "get_index_stocks": _FUNDAMENTAL_MSG.format(name="get_index_stocks"),
}


class OrderCost:                                  # 聚宽 set_order_cost 的参数对象(记录性)
    def __init__(self, **kw): self.__dict__.update(kw)


def _install_jqdata_stub():
    """让 `from jqdata import *` 成功(空导出;真实 API 由垫片注入命名空间)。"""
    import sys, types
    for name in ("jqdata", "jqlib", "jqfactor"):
        if name not in sys.modules:
            m = types.ModuleType(name); m.__all__ = []
            sys.modules[name] = m


class JQAdapter:
    """把一份聚宽策略源码包成沙箱策略:on_bar 驱动 initialize/run_daily/handle_data。"""

    def __init__(self, source, overrides=None, capital=1_000_000.0, funda=None):
        self._source = source
        self._overrides = overrides or {}
        self._capital = capital
        self._funda = funda                      # FundamentalStore(可选:解锁基本面家族)
        self._started = False
        self._weights = {}                       # 持仓目标(跨日持续,直到被改)
        self._ctx = None                         # 当前 PITContext

    # ── 垫片 API(闭包到实例)────────────────────────────
    def _px_now(self, sec):
        h = self._ctx.history(sec, 1)
        return float(h[-1]) if len(h) else float("nan")

    def _attribute_history(self, security, count, unit="1d", fields=("close",),
                           skip_paused=True, df=True, fq="pre"):
        if unit != "1d":
            raise UnsupportedJQAPI("垫片仅支持日频 unit='1d'")
        arr = self._ctx.history(security, int(count) + 1)[:-1]   # 聚宽语义:不含当日
        cols = {f: (arr if f == "close" else _unsupported(f"attribute_history field={f}")())
                for f in (fields if isinstance(fields, (list, tuple)) else [fields])}
        if not df:
            return cols                                          # 聚宽 df=False:dict[np.ndarray]
        try:
            import pandas as pd
            return pd.DataFrame(cols)
        except ImportError:
            return cols

    def _history(self, count, unit="1d", field="close", security_list=None, **k):
        secs = security_list or self._ctx.symbols
        if field != "close" or unit != "1d":
            raise UnsupportedJQAPI("垫片 history 仅支持 field='close', unit='1d'")
        data = {s: self._ctx.history(s, int(count) + 1)[:-1] for s in secs}
        try:
            import pandas as pd
            return pd.DataFrame(data)
        except ImportError:
            return data

    def _get_current_data(self):
        out = {}
        for s in self._ctx.symbols:
            try:
                op = self._ctx.history(s, 1, field="open")
                day_open = float(op[-1]) if len(op) else None
            except KeyError:
                day_open = None                      # 数据无 open 列 → 退化为 close
            out[s] = _Bar(self._px_now(s), day_open)
        return out

    def _get_bars(self, security, count, unit="1d", fields=("close",),
                  include_now=False, **k):
        """聚宽 get_bars 语义:默认不含当日;返回 numpy 结构数组(与聚宽一致,
        使 bars['close'][-1] / .mean() 原样可用)。仅支持日频 close。"""
        if unit != "1d":
            raise UnsupportedJQAPI("垫片 get_bars 仅支持日频 unit='1d'")
        flds = list(fields) if isinstance(fields, (list, tuple)) else [fields]
        if flds != ["close"]:
            raise UnsupportedJQAPI(f"垫片 get_bars 仅支持 fields=['close'],拿到 {flds}")
        n = int(count) + (0 if include_now else 1)
        arr = self._ctx.history(security, n)
        if not include_now:
            arr = arr[:-1] if len(arr) else arr
        out = np.zeros(len(arr), dtype=[("close", float)])
        out["close"] = arr
        return out

    @staticmethod
    def _no_side(kw):
        if kw.get("side") not in (None, "long"):
            raise UnsupportedJQAPI("期货双向持仓 side='short' 垫片暂未支持"
                                   "(需 long/short 两腿净额记账)——诚实拒绝,不静默按多头处理")

    def _order_target_value(self, security, value, **kw):
        self._no_side(kw)
        self._weights[security] = float(value) / self._capital

    def _order_target(self, security, amount, **kw):
        self._no_side(kw)
        self._weights[security] = float(amount) * self._px_now(security) / self._capital

    def _order_value(self, security, value, **kw):
        self._no_side(kw)
        self._weights[security] = self._weights.get(security, 0.0) + float(value) / self._capital

    def _order(self, security, amount, **kw):
        self._no_side(kw)
        self._weights[security] = (self._weights.get(security, 0.0)
                                   + float(amount) * self._px_now(security) / self._capital)

    def _today(self):
        d = self._ctx.date
        return d.item() if hasattr(d, "item") else d

    def _clamp_pit_date(self, date):
        """基本面查询日期钳制:>今日 = 前视企图,物理拒绝;≤今日 合法。"""
        if date is None:
            return self._today()
        d = datetime_parse(date)
        if d > self._today():
            raise UnsupportedJQAPI(
                f"查询未来日期的基本面数据(date={d},今日={self._today()})——"
                f"前视企图,物理拒绝")
        return d

    def _pit_index_stocks(self, index, date=None):
        return self._funda.constituents(index, self._clamp_pit_date(date))

    def _pit_fundamentals(self, q, date=None):
        return q._execute(self._funda.asof(self._clamp_pit_date(date)))

    def _run_monthly(self, func, monthday=1, *a, **kw):
        """聚宽 run_monthly:每月 monthday 号(或其后的第一个交易日)运行一次。"""
        self._monthly.append({"func": func, "monthday": int(monthday), "last": None})

    def _run_weekly(self, func, weekday=1, *a, **kw):
        """聚宽 run_weekly:每周第 weekday 个交易日(周一=1;节假日顺延)运行一次。"""
        self._weekly.append({"func": func, "weekday": int(weekday), "last": None})

    def _run_daily(self, func, *a, **kw):
        t = kw.get("time", a[0] if a else None)
        if t in ("every_bar", "every_minute"):
            # 聚宽 every_bar 跟随回测频率;垫片=日频 → 每日一次。
            # 真正依赖分钟驱动的逻辑,会在请求 '1m' 数据时被拦(不会静默错跑)。
            print(f"⚠ run_daily(time='{t}'):垫片为日频回测,该函数每日执行一次;"
                  f"若策略依赖分钟级驱动,其分钟数据请求会被明确拒绝")
        self._scheduled.append(func)

    _POBO_MARKERS = ("PoboAPI", "OnStart(", "OnBar(", "OnMarketQuotationInitialEx",
                     "QuickInsertOrder", "GetMainContract", "GetQuote(")

    def _boot(self):
        # 平台识别:喂错平台的策略必须点名拒绝,绝不静默空跑出"无 edge"的假判决
        hits = [m for m in self._POBO_MARKERS if m in self._source]
        if hits:
            raise UnsupportedJQAPI(
                f"这不是聚宽策略——检测到真格量化(POBO)API:{hits[:3]}。"
                f"POBO 事件式策略请按 MIGRATION.md 手工迁移决策核(on_bar 契约),"
                f"或等 POBO 垫片家族。静默空跑会产出错误判决,故直接拒绝。")
        _install_jqdata_stub()                     # 让 `from jqdata import *` 通过
        g = _G()
        ns = {"g": g, "log": _Log(), "np": np, "OrderCost": OrderCost,
              "attribute_history": self._attribute_history,
              "history": self._history,
              "get_bars": self._get_bars,
              "get_trades": lambda: {},            # 记账模型无逐笔成交,诚实返回空
              "send_message": lambda *a, **k: None,
              "get_current_data": self._get_current_data,
              "order_target_value": self._order_target_value,
              "order_target": self._order_target,
              "order_value": self._order_value,
              "order": self._order,
              "run_daily": self._run_daily,
              "run_monthly": self._run_monthly,
              "run_weekly": self._run_weekly,
              "set_benchmark": lambda *a, **k: None, "set_option": lambda *a, **k: None,
              "set_order_cost": lambda *a, **k: None, "set_slippage": lambda *a, **k: None,
              "set_universe": lambda *a, **k: None,
              "set_subportfolios": lambda *a, **k: None,
              "SubPortfolioConfig": lambda *a, **k: None}
        ns.update({name: _unsupported(name, msg) for name, msg in _UNSUPPORTED.items()})
        if self._funda is None:
            # 无时点仓库:查询 DSL 本身也给教育性拒绝(而非 NameError)
            for nm in ("query", "valuation", "balance", "income", "indicator"):
                ns[nm] = _unsupported(nm, _FUNDAMENTAL_MSG.format(name=nm))
        else:
            from quantros.fundamentals import make_funda_ns
            ns.update(make_funda_ns(self._funda))
            ns["get_index_stocks"] = self._pit_index_stocks
            ns["get_fundamentals"] = self._pit_fundamentals
        self._scheduled = []
        self._monthly = []
        self._weekly = []
        exec(compile(self._source, "<jq_strategy>", "exec"), ns)
        self._ns = ns
        if "initialize" not in ns and "handle_data" not in ns:
            raise UnsupportedJQAPI(
                "源码里既没有 initialize 也没有 handle_data——不像聚宽策略。"
                "静默空跑会产出'无 edge'的假判决,故直接拒绝。")
        self._jqctx = _Context(self)
        if "initialize" in ns:
            ns["initialize"](self._jqctx)
        for key, val in self._overrides.items():             # 参数族覆盖(诚实义务:全部搜索空间)
            setattr(g, key, val)
        self._started = True

    # ── 沙箱契约 ──────────────────────────────────────
    def on_bar(self, ctx):
        self._ctx = ctx
        if not self._started:
            self._boot()
        import datetime as _dt
        self._jqctx.current_dt = _dt.datetime.combine(
            ctx.date if not hasattr(ctx.date, "item") else ctx.date.item(), _dt.time(15, 0))
        for func in self._scheduled:
            func(self._jqctx)
        d = self._jqctx.current_dt.date()
        for m in self._monthly:                       # 每月 monthday 号或其后首个交易日
            if d.day >= m["monthday"] and m["last"] != (d.year, d.month):
                m["last"] = (d.year, d.month)
                m["func"](self._jqctx)
        for wk in self._weekly:                       # 每周第 weekday 个交易日(节假日顺延)
            iso = d.isocalendar()
            if d.isoweekday() >= wk["weekday"] and wk["last"] != (iso[0], iso[1]):
                wk["last"] = (iso[0], iso[1])
                wk["func"](self._jqctx)
        if "handle_data" in self._ns:
            self._ns["handle_data"](self._jqctx, self._get_current_data())
        return {s: w for s, w in self._weights.items() if s in ctx.symbols}


def run_jq_strategy(source, data, *, overrides=None, capital=1_000_000.0, bps=10.0,
                    funda=None, verbose=True):
    """零改写跑一份聚宽策略源码(字符串或文件路径)→ 沙箱结果(含 sandbox_verified)。
    funda: 可选 FundamentalStore,解锁 get_index_stocks/get_fundamentals(时点化)。"""
    if "\n" not in source and source.endswith(".py"):
        with open(source) as f:
            source = f.read()
    return run_sandbox(JQAdapter(source, overrides=overrides, capital=capital, funda=funda),
                       data, bps=bps, verbose=verbose)


def run_jq_grid(source, data, grid, *, capital=1_000_000.0, bps=10.0,
                funda=None, verbose=True):
    """参数族过沙箱:grid = [{'fast':5,'slow':20}, ...](逐组覆盖到 g)。
    返回 {配置名: 日净收益},直接喂 evaluate_returns(sandbox_verified=True)。
    ⚠️ grid 必须是你【真正搜索过的全部组合】——少报=③被高估。"""
    if "\n" not in source and source.endswith(".py"):
        with open(source) as f:
            source = f.read()
    out = {}
    for cfg in grid:
        name = ",".join(f"{k}={v}" for k, v in cfg.items())
        res = run_sandbox(JQAdapter(source, overrides=cfg, capital=capital, funda=funda),
                          data, bps=bps, verbose=False)
        out[name] = res["returns"]
        if verbose:
            print(f" [聚宽垫片网格] {name}: 净夏普={res['net_sharpe']:.2f}")
    return out
