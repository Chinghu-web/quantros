"""
from_jq 聚宽通用行情适配器回归套件 —— 假 jqdatasdk 离线钉转换/缓存/拒跑。
"""
import sys
import types
import datetime
import polars as pl
import pytest
import quantros.data as qd


def _fake_jq(monkeypatch):
    """注入假 jqdatasdk:返回两只标的、3 天的长表(含停牌剔除后的参差)。"""
    import pandas as pd
    fake = types.ModuleType("jqdatasdk")
    fake.auth = lambda u, p: None
    d0 = datetime.date(2025, 1, 6)
    rows = []
    for i in range(3):
        rows.append({"time": d0 + datetime.timedelta(days=i), "code": "510300.XSHG",
                     "close": 4.0 + i * 0.01, "money": 1e9})
        if i != 1:                                        # 000001 第二天"停牌"缺行
            rows.append({"time": d0 + datetime.timedelta(days=i), "code": "000001.XSHE",
                         "close": 10.0 + i * 0.1, "money": 5e8})
    fake.get_price = lambda *a, **k: pd.DataFrame(rows)
    monkeypatch.setitem(sys.modules, "jqdatasdk", fake)


def test_from_jq_converts_and_caches(tmp_path, monkeypatch):
    _fake_jq(monkeypatch)
    monkeypatch.setattr(qd, "JQ_CACHE_DIR", tmp_path)
    monkeypatch.setenv("JQ_USER", "u"); monkeypatch.setenv("JQ_PASS", "p")
    df = qd.from_jq(["510300.XSHG", "000001.XSHE"], "2025-01-06", "2025-01-08")
    assert set(["trading_date", "symbol", "close", "adv"]).issubset(df.columns)
    assert df["symbol"].n_unique() == 2 and df["adv"].min() > 0
    assert len(list(tmp_path.glob("jq_prices_*.parquet"))) == 1     # 已缓存

    # 第二次:删掉凭证 + 换成会爆炸的 get_price → 必须走缓存,离线成功
    monkeypatch.delenv("JQ_USER"); monkeypatch.delenv("JQ_PASS")
    sys.modules["jqdatasdk"].get_price = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("不该联网"))
    df2 = qd.from_jq(["510300.XSHG", "000001.XSHE"], "2025-01-06", "2025-01-08")
    assert df2.equals(df)


def test_from_jq_refuses_without_credentials(tmp_path, monkeypatch):
    monkeypatch.setattr(qd, "JQ_CACHE_DIR", tmp_path)
    monkeypatch.delenv("JQ_USER", raising=False)
    monkeypatch.delenv("JQ_PASS", raising=False)
    with pytest.raises(RuntimeError, match="凭证"):
        qd.from_jq(["510300.XSHG"], "2025-01-06", "2025-01-08")
