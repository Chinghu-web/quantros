"""数据接入层回归套件 —— 规范化、缺列报错、adv 估算、CSV 往返。"""
import sys
import types
import numpy as np
import pytest
import polars as pl
from quantros.data import validate, ensure_adv, load_csv, from_akshare_etf, from_akshare_stock


def _raw():
    return pl.DataFrame({
        "trading_date": ["2025-01-02", "2025-01-03", "2025-01-02", "2025-01-03"],
        "symbol": ["IF", "IF", "IH", "IH"],
        "close": [4000.0, 4010.0, 3000.0, 2990.0],
        "volume": [10000.0, 12000.0, 8000.0, 9000.0],
    }).with_columns(pl.col("trading_date").str.to_date())


def test_validate_requires_close():
    bad = _raw().drop("close")
    with pytest.raises(ValueError):
        validate(bad)


def test_ensure_adv_from_volume():
    df = ensure_adv(validate(_raw()))
    assert "adv" in df.columns
    assert df["adv"].min() > 0


def test_no_volume_no_adv_is_honest():
    """无 volume 也无 adv → 不臆造 adv 列(容量柱会据此 BLIND)。"""
    df = ensure_adv(validate(_raw().drop("volume")))
    assert "adv" not in df.columns


def test_load_csv_roundtrip(tmp_path):
    p = tmp_path / "px.csv"
    _raw().write_csv(p)
    df = load_csv(p)
    assert set(["trading_date", "symbol", "close", "adv"]).issubset(df.columns)
    assert df["trading_date"].dtype == pl.Date
    # 排序保证:同一 symbol 内日期递增
    first = df.filter(pl.col("symbol") == "IF")
    assert first["trading_date"].is_sorted()


def test_unadjusted_data_understates_returns():
    """今日核心教训固化成测试:未复权价(除息跳空)会系统性低估含分红标的收益。
    构造一条'每20天分红1%(价格向下跳)'的未复权序列 vs 其复权版,
    复权后累计收益必须显著更高——这就是全球配置策略被冤枉的根因。"""
    n = 300
    gross = np.full(n, 1.0005)                    # 每日真实总回报 +0.05%
    div_days = np.arange(20, n, 20)
    unadj = gross.copy()
    unadj[div_days] -= 0.01                        # 除息日:未复权价额外向下跳 1%(分红)
    adj_cum = np.prod(gross)                       # 复权(含分红)累计
    unadj_cum = np.prod(unadj)                     # 未复权累计(丢了分红)
    assert adj_cum > unadj_cum * 1.10             # 未复权低估 >10%(15次分红×1%)


def _fake_akshare(monkeypatch, capture):
    import pandas as pd
    fake = types.ModuleType("akshare")
    def _hist(symbol, period, adjust, start_date, end_date, **k):
        capture["adjust"] = adjust; capture["symbol"] = symbol
        return pd.DataFrame({"日期": ["2020-01-02", "2020-01-03"], "收盘": [10.0, 10.1],
                             "开盘": [9.9, 10.0], "最高": [10.2, 10.2], "最低": [9.8, 10.0],
                             "成交额": [1e8, 1e8]})
    fake.stock_zh_a_hist = _hist
    fake.fund_etf_hist_em = _hist
    monkeypatch.setitem(sys.modules, "akshare", fake)


def test_from_akshare_etf_defaults_hfq(monkeypatch):
    """官方 ETF 适配器默认后复权;拒绝未复权。"""
    cap = {}
    _fake_akshare(monkeypatch, cap)
    df = from_akshare_etf(["510880.XSHG"])
    assert cap["adjust"] == "hfq"                  # 默认后复权
    assert set(["trading_date", "symbol", "close", "adv"]).issubset(df.columns)
    with pytest.raises(ValueError, match="复权"):
        from_akshare_etf(["510880"], adjust="")   # 未复权被拒


def test_from_akshare_stock_defaults_hfq(monkeypatch):
    cap = {}
    _fake_akshare(monkeypatch, cap)
    from_akshare_stock(["000001.XSHE"])
    assert cap["adjust"] == "hfq" and cap["symbol"] == "000001"   # 自动取6位代码
    with pytest.raises(ValueError, match="复权"):
        from_akshare_stock(["000001"], adjust="none")
