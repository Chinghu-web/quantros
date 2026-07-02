"""
QuantROS 基本面家族 (Fundamentals Family) —— 时点化成分股 + 公告日对齐财报。

解锁 get_index_stocks / get_fundamentals 类策略(格雷厄姆选股/小市值/低PB轮动),
同时物理堵死基本面回测的两大病:
  · 幸存者偏差:成分股按【当日快照】返回(月度快照,取 ≤当日 最近一期);
  · 财报前视:面板数据来自聚宽 get_fundamentals(date=快照日)——聚宽该接口
    按【公告日】返回当日可见的最近一期财报;本地查询只允许 ≤当日,
    查询未来日期 = 前视企图,直接拒绝。

组成:
  FundamentalStore              时点面板仓库(成分股快照 + 因子面板,parquet 可存取)
  generate_fundamental_market   合成市场(价格 + 面板 + 轮动成分),测试/演示用
  fetch_fundamental_store       聚宽真实取数(凭证进终端,本地缓存,额度只花一次)
  query/valuation/balance DSL   聚宽查询语法仿真 → 策略零改写(见 jqcompat 接线)

⚠️ 诚实边界:
  · 快照频率=月度(约21交易日):月内的成分调整/财报发布有最长一个月的滞后
    ——滞后方向是【保守】的(晚知道,不是提前知道),不引入前视;
  · 面板字段以取数时声明的为准,策略用到面板外字段 → 点名拒绝;
  · 聚宽公告日对齐的正确性依赖数据商口径,平台不重复审计。
"""
import datetime
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import polars as pl

CACHE_DIR = Path(__file__).resolve().parent.parent / "data_cache"

DEFAULT_FIELDS = ["pb_ratio", "market_cap", "total_assets", "total_liability",
                  "total_current_assets", "total_current_liability"]


class FundamentalStore:
    """时点面板仓库。panel: (snap_date, code, 字段...);members: (snap_date, code)。"""

    def __init__(self, index, panel: pl.DataFrame, members: pl.DataFrame):
        self.index = index
        self.panel = panel.sort(["snap_date", "code"])
        self.members = members.sort(["snap_date", "code"])
        self._snaps = np.sort(self.panel["snap_date"].unique().to_numpy())
        self.fields = [c for c in panel.columns if c not in ("snap_date", "code")]

    def _last_snap(self, d):
        k = int(np.searchsorted(self._snaps, np.datetime64(d), side="right"))
        return None if k == 0 else self._snaps[k - 1]

    def asof(self, d) -> pl.DataFrame:
        """≤d 最近一期面板切片(code + 字段);无 → 空表。物理上到不了未来。"""
        s = self._last_snap(d)
        if s is None:
            return self.panel.clear()
        return self.panel.filter(pl.col("snap_date") == s).drop("snap_date")

    def constituents(self, index, d):
        if index != self.index:
            raise KeyError(f"本仓库为指数 {self.index} 构建,策略请求 {index};"
                           f"请为该指数另建 store")
        s = self._last_snap(d)
        if s is None:
            return []
        return self.members.filter(pl.col("snap_date") == s)["code"].to_list()

    # ── 持久化 ────────────────────────────────────────
    def save(self, key):
        CACHE_DIR.mkdir(exist_ok=True)
        self.panel.write_parquet(CACHE_DIR / f"funda_panel_{key}.parquet")
        self.members.write_parquet(CACHE_DIR / f"funda_members_{key}.parquet")

    @classmethod
    def load(cls, index, key):
        return cls(index,
                   pl.read_parquet(CACHE_DIR / f"funda_panel_{key}.parquet"),
                   pl.read_parquet(CACHE_DIR / f"funda_members_{key}.parquet"))


# ── 合成市场(机制验证/演示)────────────────────────────
def generate_fundamental_market(n_stocks=30, n_days=500, index="000300.XSHG",
                                snap_every=21, seed=7):
    """价格面板 + 时点基本面 + 轮动成分股。
    market_cap 以【亿】计(对齐聚宽 valuation.market_cap 单位),约 15~45 亿区间,
    使 between(20,30) 这类模板条件真实可命中;成分 = 每期市值最大的 2/3(会轮动)。"""
    rng = np.random.default_rng(seed)
    start = datetime.date(2024, 1, 2)
    dates = []
    d, i = start, 0
    while len(dates) < n_days:
        d = start + datetime.timedelta(days=i); i += 1
        if d.weekday() < 5:
            dates.append(d)
    codes = [f"{600000+j}.XSHG" for j in range(n_stocks)]
    shares = rng.uniform(5e8, 15e8, n_stocks)                 # 股本
    book = rng.uniform(1.5, 4.0, n_stocks)                    # 每股净资产
    px0 = rng.uniform(2.0, 5.0, n_stocks)
    rets = rng.normal(0.0003, 0.015, (n_days, n_stocks))
    px = px0 * np.cumprod(1 + rets, axis=0)

    prices = pl.DataFrame({
        "trading_date": pl.Series([d for d in dates for _ in codes]).cast(pl.Date),
        "symbol": codes * n_days,
        "close": px.ravel(),
    })

    snap_idx = list(range(0, n_days, snap_every))
    p_rows, m_rows = [], []
    for t in snap_idx:
        mc = px[t] * shares / 1e8                             # 市值(亿)
        pb = px[t] / book
        ta = mc * rng.uniform(2.0, 4.0, n_stocks)             # 总资产等(量级合理即可)
        tl = ta * rng.uniform(0.3, 0.8, n_stocks)
        tca = ta * rng.uniform(0.2, 0.5, n_stocks)
        tcl = tca / rng.uniform(0.8, 2.0, n_stocks)
        for j, c in enumerate(codes):
            p_rows.append({"snap_date": dates[t], "code": c, "pb_ratio": float(pb[j]),
                           "market_cap": float(mc[j]), "total_assets": float(ta[j]),
                           "total_liability": float(tl[j]),
                           "total_current_assets": float(tca[j]),
                           "total_current_liability": float(tcl[j])})
        member_j = np.argsort(mc)[-(n_stocks * 2 // 3):]      # 市值前 2/3 入指数 → 轮动
        for j in member_j:
            m_rows.append({"snap_date": dates[t], "code": codes[j]})
    store = FundamentalStore(index, pl.DataFrame(p_rows), pl.DataFrame(m_rows))
    return prices, store


# ── 聚宽真实取数(凭证进终端,缓存优先)────────────────────
def fetch_fundamental_store(index="000300.XSHG", start="2020-01-01", end="2025-12-31",
                            fields=None, snap_every=21, refresh=False):
    """按月度快照拉:成分股(get_index_stocks(date=snap))+ 面板
    (get_fundamentals(query(...), date=snap),聚宽按公告日返回当日可见财报)。
    额度 ≈ 2×快照数(十年约 240 次调用),缓存后离线。"""
    fields = fields or DEFAULT_FIELDS
    key = hashlib.md5(json.dumps([index, str(start), str(end), sorted(fields), snap_every],
                                 ensure_ascii=False).encode()).hexdigest()[:12]
    try:
        return FundamentalStore.load(index, key)
    except FileNotFoundError:
        pass
    if refresh is False and not (os.environ.get("JQ_USER") and os.environ.get("JQ_PASS")):
        raise RuntimeError("首次构建基本面仓库需聚宽凭证(之后走本地缓存):\n"
                           "  JQ_USER=手机号 JQ_PASS=密码 python3 你的脚本.py")
    import jqdatasdk as jq
    jq.auth(str(os.environ["JQ_USER"]), str(os.environ["JQ_PASS"]))
    from jqdatasdk import query, valuation, balance
    cal = jq.get_price("000001.XSHG", start_date=str(start), end_date=str(end),
                       frequency="daily", fields=["close"], panel=False)
    idx = cal.index if cal.index.name or str(cal.index.dtype).startswith("datetime") else cal["time"]
    days = sorted(pl.Series("d", np.asarray(idx.values)).cast(pl.Date).to_list())
    snaps = days[::snap_every]
    val_cols = [getattr(valuation, f) for f in fields if hasattr(valuation, f)]
    bal_cols = [getattr(balance, f) for f in fields if hasattr(balance, f)]
    p_rows, m_rows = [], []
    for snap in snaps:
        codes = jq.get_index_stocks(index, date=snap)
        m_rows += [{"snap_date": snap, "code": c} for c in codes]
        q = query(valuation.code, *val_cols, *bal_cols).filter(valuation.code.in_(codes))
        df = jq.get_fundamentals(q, date=snap)                # 聚宽:按公告日可见
        for r in df.itertuples(index=False):
            row = {"snap_date": snap, "code": r.code}
            row.update({f: float(getattr(r, f)) if getattr(r, f) is not None else float("nan")
                        for f in fields if hasattr(r, f)})
            p_rows.append(row)
        print(f"  快照 {snap}: 成分 {len(codes)} 只")
    store = FundamentalStore(index, pl.DataFrame(p_rows), pl.DataFrame(m_rows))
    store.save(key)
    return store


# ── 聚宽查询 DSL 仿真(零改写的关键)─────────────────────
class _Col:
    def __init__(self, table, name): self.table, self.name = table, name
    def __lt__(self, v): return ("lt", self.name, v)
    def __le__(self, v): return ("le", self.name, v)
    def __gt__(self, v): return ("gt", self.name, v)
    def __ge__(self, v): return ("ge", self.name, v)
    def in_(self, xs): return ("in", self.name, list(xs))
    def between(self, a, b): return ("between", self.name, (a, b))
    def __truediv__(self, other): return _Ratio(self, other)
    def asc(self): return ("asc", self.name)
    def desc(self): return ("desc", self.name)


class _Ratio:                                   # 支持 balance.a/balance.b > 1.2 这类过滤
    def __init__(self, a, b): self.a, self.b = a.name, b.name
    def __gt__(self, v): return ("ratio_gt", (self.a, self.b), v)
    def __lt__(self, v): return ("ratio_lt", (self.a, self.b), v)


class _Table:
    def __init__(self, name, available):
        self._name, self._avail = name, set(available) | {"code"}

    def __getattr__(self, item):
        if item.startswith("_"):
            raise AttributeError(item)
        if item not in self._avail:
            from quantros.jqcompat import UnsupportedJQAPI
            raise UnsupportedJQAPI(
                f"字段 {self._name}.{item} 不在时点面板中(可用: {sorted(self._avail)});"
                f"重建仓库时用 fields=[...] 声明所需字段——面板外字段拒绝,不猜")
        return _Col(self._name, item)


_OPS = {"lt": "<", "le": "<=", "gt": ">", "ge": ">="}


class _Query:
    def __init__(self, *cols):
        self._cols = [c.name for c in cols]
        self._filters, self._order = [], None

    def filter(self, *specs):
        self._filters += list(specs); return self

    def order_by(self, spec):
        self._order = spec; return self

    def _execute(self, slice_df: pl.DataFrame):
        df = slice_df
        for spec in self._filters:
            kind = spec[0]
            if kind == "in":
                df = df.filter(pl.col(spec[1]).is_in(spec[2]))
            elif kind == "between":
                a, b = spec[2]; df = df.filter(pl.col(spec[1]).is_between(a, b))
            elif kind in _OPS:
                df = df.filter(eval(f"pl.col(spec[1]) {_OPS[kind]} spec[2]"))
            elif kind == "ratio_gt":
                a, b = spec[1]; df = df.filter(pl.col(a) / pl.col(b) > spec[2])
            elif kind == "ratio_lt":
                a, b = spec[1]; df = df.filter(pl.col(a) / pl.col(b) < spec[2])
            else:
                from quantros.jqcompat import UnsupportedJQAPI
                raise UnsupportedJQAPI(f"查询过滤 {kind} 未支持")
        if self._order:
            kind, name = self._order
            df = df.sort(name, descending=(kind == "desc"))
        cols = list(dict.fromkeys((["code"] if "code" not in self._cols else []) + self._cols))
        return df.select([c for c in cols if c in df.columns]).to_pandas()


def make_funda_ns(store: FundamentalStore):
    """给 jqcompat 注入的命名空间:query/valuation/balance(仅面板内字段可用)。"""
    return {
        "query": _Query,
        "valuation": _Table("valuation", [f for f in store.fields]),
        "balance": _Table("balance", [f for f in store.fields]),
        "indicator": _Table("indicator", []),          # 面板未含 → 用即点名拒绝
        "income": _Table("income", []),
    }
