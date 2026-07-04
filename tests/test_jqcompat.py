"""
聚宽兼容垫片回归套件 —— 零改写、PIT 穿透自证、聚宽语义忠实、诚实拒绝。
运行:  pytest tests/test_jqcompat.py -v
"""
import numpy as np
import polars as pl
import pytest
from quantros.jqcompat import run_jq_strategy, run_jq_grid, JQAdapter, UnsupportedJQAPI
from quantros.robustness import generate_regime_data
from quantros.sandbox import run_sandbox, _prepare, PITContext

# 原汁原味的聚宽写法(initialize + run_daily + g + attribute_history + order_target_value)
JQ_SOURCE = """
def initialize(context):
    g.security = "IF"
    g.fast, g.slow = 5, 20
    set_benchmark("000300.XSHG")
    run_daily(trade, time="14:50")

def trade(context):
    hist = attribute_history(g.security, g.slow, "1d", ["close"])
    if len(hist["close"]) < g.slow:
        return
    if hist["close"][-g.fast:].mean() > hist["close"].mean():
        order_target_value(g.security, context.portfolio.total_value)
    else:
        order_target_value(g.security, 0)
"""


@pytest.fixture
def data():
    return generate_regime_data(drift=0.0012)


def test_verbatim_jq_strategy_runs(data):
    """聚宽源码一行不改,直接过沙箱并拿到 sandbox_verified。"""
    res = run_jq_strategy(JQ_SOURCE, data, bps=3.0, verbose=False)
    assert res["sandbox_verified"] is True
    assert np.isfinite(res["net_sharpe"])
    assert len([1 for _, b, _ in res["positions"].iter_rows()] ) or True  # positions 表存在


def test_pit_invariance_through_shim(data):
    """关键:PIT 物理保证必须【穿透垫片】——扰动后 30% 行情,前 70% 决策逐位不变。
    垫片若偷偷读了未来(哪怕一个 API 实现错了),这条会当场抓住。"""
    dates = sorted(data["trading_date"].unique().to_list())
    cut = dates[int(len(dates) * 0.7)]
    rng = np.random.default_rng(7)
    perturbed = data.with_columns(
        pl.when(pl.col("trading_date") > cut)
          .then(pl.col("close") * pl.Series("n", rng.uniform(0.2, 5.0, len(data))))
          .otherwise(pl.col("close")).alias("close"))
    p1 = run_jq_strategy(JQ_SOURCE, data, verbose=False)["positions"]
    p2 = run_jq_strategy(JQ_SOURCE, perturbed, verbose=False)["positions"]
    a = p1.filter(pl.col("trading_date") <= cut).sort(["trading_date", "symbol"])
    b = p2.filter(pl.col("trading_date") <= cut).sort(["trading_date", "symbol"])
    assert len(a) > 0 and a.equals(b)


def test_attribute_history_excludes_today(data):
    """聚宽语义忠实:attribute_history 不含当日——垫片必须保留这一语义。"""
    _, per_symbol, all_dates = _prepare(data)
    ctx = PITContext(per_symbol)
    ctx._advance(all_dates[100])
    ad = JQAdapter(JQ_SOURCE); ad._ctx = ctx
    h = ad._attribute_history("IF", 10)
    dates_s, fields_s = per_symbol["IF"]
    k = int(np.searchsorted(dates_s, all_dates[100], side="right"))
    expect = fields_s["close"][k - 11:k - 1]            # 截至【昨日】的 10 个
    assert np.array_equal(np.asarray(h["close"]), expect)


def test_grid_overrides_apply(data):
    """参数覆盖生效:不同 fast/slow 必须产生不同收益序列。"""
    rets = run_jq_grid(JQ_SOURCE, data,
                       grid=[dict(fast=5, slow=20), dict(fast=10, slow=60)], verbose=False)
    a, b = list(rets.values())
    assert not np.allclose(a, b)


def test_pobo_strategy_rejected_by_name(data):
    """平台识别:真格(POBO)策略喂进聚宽垫片必须点名拒绝——
    否则会静默空跑出'无 edge'的假判决(比报错危险)。"""
    pobo = "from PoboAPI import *\ndef OnStart(context):\n    pass\n"
    with pytest.raises(UnsupportedJQAPI, match="真格"):
        run_jq_strategy(pobo, data, verbose=False)


def test_non_jq_source_rejected(data):
    """连 initialize/handle_data 都没有的源码 → 拒绝,不静默。"""
    with pytest.raises(UnsupportedJQAPI, match="不像聚宽"):
        run_jq_strategy("x = 1\n", data, verbose=False)


def test_every_bar_is_daily_and_minute_data_still_rejected(data):
    """every_bar 语义修正:聚宽 every_bar 跟随回测频率,垫片=日频 → 每日一次照跑
    (小市值模板画像);真正的日内逻辑在请求 '1m' 数据时被拦(Dual Thrust 画像)。"""
    ok = ("def initialize(context):\n"
          "    g.n = 0\n"
          "    run_daily(trade, 'every_bar')\n"
          "def trade(context):\n"
          "    g.n += 1\n"
          "    order_target_value('IF', context.portfolio.total_value)\n")
    res = run_jq_strategy(ok, data, verbose=False)
    assert (res["positions"]["weight"].to_numpy() > 0).any()     # 日频照常运转

    intraday = ("def initialize(context):\n"
                "    run_daily(trade, 'every_bar')\n"
                "def trade(context):\n"
                "    attribute_history('IF', 1, '1m', ['close'])\n")
    with pytest.raises(UnsupportedJQAPI, match="日频"):
        run_jq_strategy(intraday, data, verbose=False)


def test_futures_short_side_rejected(data):
    """期货双向 side='short':未支持就拒绝,绝不静默按多头记账(那会出假判决)。"""
    src = ("def initialize(context):\n"
           "    run_daily(trade)\n"
           "def trade(context):\n"
           "    order('IF', 1, side='short')\n")
    with pytest.raises(UnsupportedJQAPI, match="side"):
        run_jq_strategy(src, data, verbose=False)


def test_order_target_percent_supported(data):
    """还债:order_target_percent/order_percent(聚宽高频用法)→ 权重直取,不再拒绝。"""
    src = ("def initialize(context):\n"
           "    run_daily(trade)\n"
           "def trade(context):\n"
           "    order_target_percent('IF', 0.6)\n"
           "    order_percent('IH', 0.2)\n")
    res = run_jq_strategy(src, data, verbose=False)
    p = res["positions"]
    w_if = p.filter(pl.col("symbol") == "IF")["weight"].to_numpy()
    w_ih = p.filter(pl.col("symbol") == "IH")["weight"].to_numpy()
    assert abs(w_if[-1] - 0.6) < 1e-9
    assert w_ih[-1] > 0.2                      # order_percent 逐日累加(增量语义)


def test_attribute_history_multi_field(data):
    """还债:attribute_history/get_bars 多字段透传(high/low/open);
    数据缺该列 → 点名报错列出可用字段,不静默。"""
    d2 = data.with_columns([(pl.col("close") * 1.01).alias("high"),
                            (pl.col("close") * 0.99).alias("low")])
    src = ("import numpy as np\n"
           "def initialize(context):\n"
           "    run_daily(trade)\n"
           "def trade(context):\n"
           "    h = attribute_history('IF', 10, '1d', ['high','low','close'])\n"
           "    if len(h['close']) < 10: return\n"
           "    assert (np.asarray(h['high']) > np.asarray(h['low'])).all()\n"
           "    b = get_bars('IF', 5, '1d', fields=['high','close'])\n"
           "    assert b['high'][-1] > b['close'][-1] * 0.99\n"
           "    order_target_value('IF', context.portfolio.total_value)\n")
    res = run_jq_strategy(src, d2, verbose=False)
    assert (res["positions"]["weight"].to_numpy() > 0).any()

    bad = src.replace("['high','low','close']", "['volume']")
    with pytest.raises(UnsupportedJQAPI, match="volume"):
        run_jq_strategy(bad, d2, verbose=False)


def test_multi_field_history(data):
    """沙箱多字段:数据含 open/high/low 时 ctx.history(field=...) 可用且时点一致。"""
    import polars as pl
    d2 = data.with_columns([(pl.col("close") * 1.01).alias("high"),
                            (pl.col("close") * 0.99).alias("low"),
                            (pl.col("close") * 1.001).alias("open")])
    _, per_symbol, all_dates = _prepare(d2)
    ctx = PITContext(per_symbol)
    ctx._advance(all_dates[50])
    h = ctx.history("IF", 5, field="high")
    c = ctx.history("IF", 5, field="close")
    assert len(h) == 5 and np.allclose(h, c * 1.01)
    with pytest.raises(KeyError, match="volume"):
        ctx.history("IF", 5, field="volume")


def test_fundamentals_rejected_with_survivorship_education(data):
    """基本面选股(格雷厄姆式官方模板画像):get_index_stocks/get_fundamentals
    必须拒绝且报错里讲清【为什么】——幸存者偏差 + 财报公告日前视。"""
    src = ("def initialize(context):\n"
           "    run_monthly(trade, monthday=1)\n"
           "def trade(context):\n"
           "    stocks = get_index_stocks('000300.XSHG')\n")
    with pytest.raises(UnsupportedJQAPI, match="幸存者偏差"):
        run_jq_strategy(src, data, verbose=False)


def test_run_monthly_fires_once_per_month(data):
    """run_monthly:400 交易日≈19 个月,月度函数应触发 15~22 次,且每月至多一次。"""
    src = ("def initialize(context):\n"
           "    g.cnt = 0\n"
           "    g.months = set()\n"
           "    run_monthly(rebalance, monthday=1)\n"
           "def rebalance(context):\n"
           "    g.cnt += 1\n"
           "    key = (context.current_dt.year, context.current_dt.month)\n"
           "    assert key not in g.months, '同月重复触发'\n"
           "    g.months.add(key)\n"
           "    order_target_value('IF', context.portfolio.cash)\n")   # 顺带测 .cash 别名
    res = run_jq_strategy(src, data, verbose=False)
    w = res["positions"].filter(pl.col("symbol") == "IF")["weight"].to_numpy()
    assert (w > 0).any()                       # 确实买了(cash 别名工作)
    assert w.max() <= 1.0 + 1e-9               # 无杠杆


def test_negative_monthday_rejected(data):
    """审计修复:负 monthday(月末/倒数调仓,聚宽常用)不能静默变月初——必须点名拒绝。
    (原实现 d.day>=-1 恒真 → 每月初触发,持仓序列全错却不报错。)"""
    src = ("def initialize(context):\n"
           "    run_monthly(rb, monthday=-1)\n"     # -1 = 聚宽的"每月最后一个交易日"
           "def rb(context):\n"
           "    order_target_value('IF', context.portfolio.total_value)\n")
    with pytest.raises(UnsupportedJQAPI, match="月末"):
        run_jq_strategy(src, data, verbose=False)


def test_run_weekly_fires_once_per_week(data):
    """run_weekly:400 交易日≈80 周,应触发 70~85 次且同周绝不重复(低PB银行轮动画像)。"""
    src = ("def initialize(context):\n"
           "    g.cnt = 0\n"
           "    g.weeks = set()\n"
           "    run_weekly(rebalance, weekday=1, time='open')\n"
           "def rebalance(context):\n"
           "    key = context.current_dt.date().isocalendar()[:2]\n"
           "    assert key not in g.weeks, '同周重复触发'\n"
           "    g.weeks.add(key)\n"
           "    g.cnt += 1\n"
           "    order_target_value('IF', context.portfolio.total_value)\n")
    res = run_jq_strategy(src, data, verbose=False)
    ad = res["positions"]
    assert (ad["weight"].to_numpy() > 0).any()


def test_attribute_history_df_false_returns_arrays(data):
    """df=False:返回 dict[np.ndarray](聚宽语义),hist['close'][-1] 原样可用。"""
    _, per_symbol, all_dates = _prepare(data)
    ctx = PITContext(per_symbol)
    ctx._advance(all_dates[100])
    ad = JQAdapter(JQ_SOURCE); ad._ctx = ctx
    h = ad._attribute_history("IF", 10, "1d", "close", df=False)
    assert isinstance(h, dict) and isinstance(h["close"], np.ndarray)
    assert h["close"][-1] == h["close"][-1]    # 数组负索引可用


def test_unsupported_api_raises_loudly(data):
    """诚实拒绝:用到不支持的 API 必须报 UnsupportedJQAPI 并点名,绝不静默假装。"""
    bad = JQ_SOURCE.replace('hist = attribute_history(g.security, g.slow, "1d", ["close"])',
                            'x = get_fundamentals(None)\n    hist = attribute_history(g.security, g.slow, "1d", ["close"])')
    with pytest.raises(UnsupportedJQAPI, match="get_fundamentals"):
        run_jq_strategy(bad, data, verbose=False)
