"""数据接入层回归套件 —— 规范化、缺列报错、adv 估算、CSV 往返。"""
import pytest
import polars as pl
from quantros.data import validate, ensure_adv, load_csv


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
