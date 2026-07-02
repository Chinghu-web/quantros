"""
基本面家族回归套件 —— 时点可见性、成分轮动、未来查询物理拒绝、DSL 正确性、
扰动自证(基本面版)、模板端到端。
运行:  pytest tests/test_fundamentals.py -v
"""
import datetime
import numpy as np
import polars as pl
import pytest
from quantros.fundamentals import generate_fundamental_market, FundamentalStore
from quantros.jqcompat import run_jq_strategy, UnsupportedJQAPI


@pytest.fixture(scope="module")
def market():
    return generate_fundamental_market()


SMALLCAP = """
def initialize(context):
    g.stocknum = 3; g.days = 0; g.refresh_rate = 5
    run_daily(trade, 'open')
def check_stocks(context):
    q = query(valuation.code, valuation.market_cap
        ).filter(valuation.market_cap.between(20,30)
        ).order_by(valuation.market_cap.asc())
    df = get_fundamentals(q)
    return list(df['code'])[:g.stocknum]
def trade(context):
    if g.days % g.refresh_rate == 0:
        for stock in list(context.portfolio.positions.keys()):
            order_target_value(stock, 0)
        picks = check_stocks(context)
        if picks:
            cash = context.portfolio.cash / len(picks)
            for stock in picks:
                order_value(stock, cash)
        g.days = 1
    else:
        g.days += 1
"""


def test_asof_is_point_in_time(market):
    """时点可见性:asof(d) = ≤d 最近一期快照;首个快照日前 = 空(不臆造)。"""
    prices, store = market
    snaps = sorted(store.panel["snap_date"].unique().to_list())
    assert len(store.asof(snaps[0] - datetime.timedelta(days=1))) == 0
    mid = snaps[3] + datetime.timedelta(days=5)          # 两期快照之间
    got = store.asof(mid)
    expect = store.panel.filter(pl.col("snap_date") == snaps[3]).drop("snap_date")
    assert got.sort("code").equals(expect.sort("code"))


def test_constituents_rotate_and_are_pit(market):
    """成分轮动:不同期成分应有差异(幸存者偏差的解药正是这份轮动名单)。"""
    _, store = market
    snaps = sorted(store.members["snap_date"].unique().to_list())
    first = set(store.constituents(store.index, snaps[0]))
    last = set(store.constituents(store.index, snaps[-1]))
    assert first != last                                  # 名单确实随时间变
    with pytest.raises(KeyError, match="另建"):
        store.constituents("399951.XSHE", snaps[0])       # 错指数点名拒绝


def test_future_fundamental_query_physically_rejected(market):
    """反作弊:策略在 t 日查询未来日期的财报/成分 → 前视企图,物理拒绝。"""
    prices, store = market
    src = ("def initialize(context):\n"
           "    run_daily(trade, 'open')\n"
           "def trade(context):\n"
           "    get_index_stocks('000300.XSHG', date='2030-01-01')\n")
    with pytest.raises(UnsupportedJQAPI, match="前视企图"):
        run_jq_strategy(src, prices, funda=store, verbose=False)


def test_query_dsl_filters_and_order(market):
    """DSL 正确性:between/in_/比值过滤/order_by 与手算一致(格雷厄姆模板画像)。"""
    _, store = market
    from quantros.fundamentals import make_funda_ns
    ns = make_funda_ns(store)
    q, val, bal = ns["query"], ns["valuation"], ns["balance"]
    d = sorted(store.panel["snap_date"].unique().to_list())[5]
    sl = store.asof(d)

    out = q(val.code, val.pb_ratio).filter(
        val.pb_ratio < 2,
        bal.total_current_assets / bal.total_current_liability > 1.2,
    ).order_by(val.pb_ratio.asc())._execute(sl)
    manual = (sl.filter((pl.col("pb_ratio") < 2)
                        & (pl.col("total_current_assets") / pl.col("total_current_liability") > 1.2))
                .sort("pb_ratio"))
    assert list(out["code"]) == manual["code"].to_list()
    assert (np.diff(out["pb_ratio"].to_numpy()) >= 0).all()

    with pytest.raises(UnsupportedJQAPI, match="不在时点面板"):
        _ = ns["income"].net_profit                       # 面板外字段点名拒绝


def test_fundamental_perturbation_cannot_change_past(market):
    """沙箱自证·基本面版:把【未来快照】的市值/PB 全部打乱,
    扰动前的持仓决策必须逐位不变——基本面通路同样喂不出未来。"""
    prices, store = market
    snaps = sorted(store.panel["snap_date"].unique().to_list())
    cut = snaps[len(snaps) * 7 // 10]
    rng = np.random.default_rng(4)
    noisy = store.panel.with_columns([
        pl.when(pl.col("snap_date") > cut)
          .then(pl.col(c) * pl.Series(rng.uniform(0.1, 9.0, len(store.panel))))
          .otherwise(pl.col(c)).alias(c)
        for c in ("market_cap", "pb_ratio")])
    store2 = FundamentalStore(store.index, noisy, store.members)
    p1 = run_jq_strategy(SMALLCAP, prices, funda=store, verbose=False)["positions"]
    p2 = run_jq_strategy(SMALLCAP, prices, funda=store2, verbose=False)["positions"]
    a = p1.filter(pl.col("trading_date") <= cut).sort(["trading_date", "symbol"])
    b = p2.filter(pl.col("trading_date") <= cut).sort(["trading_date", "symbol"])
    assert len(a) > 0 and a.equals(b)


def test_smallcap_template_end_to_end(market):
    """端到端:小市值模板(真实 query 语法)在时点仓库上运转并真实持仓、无杠杆。"""
    prices, store = market
    res = run_jq_strategy(SMALLCAP, prices, funda=store, bps=15.0, verbose=False)
    w = res["positions"]["weight"].to_numpy()
    daily = res["positions"].group_by("trading_date").agg(pl.col("weight").sum())
    assert (daily["weight"].to_numpy() > 0).any()          # 真的买了
    assert daily["weight"].max() <= 1.0 + 1e-9             # 无杠杆
    assert res["sandbox_verified"] is True
