"""
聚宽桥回归套件 —— 下游全路径(缓存→网格→冠军解析→④→判决)离线钉死。
真实取数需凭证,无法在 CI 跑;此处用 ETF 量级合成缓存验证"数据一到即直通"。
"""
import polars as pl
import pytest
import quantros.jq_bridge as jb
from quantros.options import generate_option_market


@pytest.fixture
def etf_cache(tmp_path, monkeypatch):
    """ETF 量级(S≈5元)合成市写入临时缓存,monkeypatch 桥常量。"""
    und, chain = generate_option_market(n_days=200, strike_step=100.0)
    und = und.with_columns((pl.col("close") / 1000).alias("close"))
    chain = chain.with_columns([(pl.col("close") / 1000).alias("close"),
                                (pl.col("strike") / 1000).alias("strike")])
    monkeypatch.setattr(jb, "CHAIN_PQ", tmp_path / "chain.parquet")
    monkeypatch.setattr(jb, "UND_PQ", tmp_path / "und.parquet")
    chain.write_parquet(jb.CHAIN_PQ); und.write_parquet(jb.UND_PQ)


def test_bridge_full_path_offline(etf_cache, capsys):
    """缓存在位 → main() 不联网直通五门,返回合法分级判决。"""
    v = jb.main()
    assert v["tier"] in {"falsified", "edge_only", "risk_falsified", "not_falsified"}
    assert "① 沙箱(诚实)" in v["gates"]
    out = capsys.readouterr().out
    assert "使用缓存" in out          # 走的是离线缓存分支


def test_bridge_refuses_without_credentials(tmp_path, monkeypatch):
    """无缓存且无凭证:给出明确指引并退出,绝不半途假跑。"""
    monkeypatch.setattr(jb, "CHAIN_PQ", tmp_path / "nope.parquet")
    monkeypatch.setattr(jb, "UND_PQ", tmp_path / "nope2.parquet")
    monkeypatch.delenv("JQ_USER", raising=False)
    monkeypatch.delenv("JQ_PASS", raising=False)
    with pytest.raises(SystemExit):
        jb.load_or_fetch()
