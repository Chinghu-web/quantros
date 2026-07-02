"""
QuantROS 数据接入层 —— 把任意来源的行情统一成五柱引擎吃的规范 schema。

规范 schema(长表,按 [symbol, trading_date] 排序):
    trading_date : Date      必需
    symbol       : str        必需
    close        : float      必需
    adv          : float      容量柱需要(美元/元日均成交额);缺则由 volume×close 估算,再缺则容量柱 BLIND
    volume       : float      可选(用于估算 adv)

任何来源(CSV / akshare / tushare / 自有库)只要落到这个 schema,五柱与中档都能直接跑。
"""
from pathlib import Path

import polars as pl

REQUIRED = ["trading_date", "symbol", "close"]
JQ_CACHE_DIR = Path(__file__).resolve().parent.parent / "data_cache"


def validate(df: pl.DataFrame) -> pl.DataFrame:
    """校验必需列、规整类型并排序。缺列直接报错,绝不静默放行。"""
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"行情缺必需列: {missing};规范 schema 见 quantros.data 文档")
    df = df.with_columns([
        pl.col("trading_date").cast(pl.Date),
        pl.col("symbol").cast(pl.Utf8),
        pl.col("close").cast(pl.Float64),
    ])
    return df.sort(["symbol", "trading_date"])


def ensure_adv(df: pl.DataFrame, window: int = 20) -> pl.DataFrame:
    """补全 adv(日均成交额):
      已有 adv → 原样;否则有 volume → 用 (volume×close) 的 window 日滚动均值估算;
      都没有 → 不加 adv 列(容量柱会诚实地 BLIND)。"""
    if "adv" in df.columns:
        return df
    if "volume" not in df.columns:
        return df
    df = df.with_columns((pl.col("volume") * pl.col("close")).alias("_dv"))
    df = df.with_columns(
        pl.col("_dv").rolling_mean(window, min_samples=1).over("symbol").alias("adv"))
    return df.drop("_dv")


def load_csv(path, window: int = 20) -> pl.DataFrame:
    """从 CSV 读行情并规范化。列至少含 trading_date,symbol,close(可含 volume/adv)。"""
    df = pl.read_csv(path, try_parse_dates=True)
    return ensure_adv(validate(df), window=window)


# ── 真实数据适配器(可选依赖,联网)──────────────────────
# 中国股指期货主力连续:IF(沪深300) IH(上证50) IC(中证500) IM(中证1000)
CN_INDEX_FUTURES = ["IF", "IH", "IC", "IM"]


def from_jq(securities, start, end, adv_window: int = 20, refresh: bool = False) -> pl.DataFrame:
    """聚宽日线(股票/ETF/指数/基金)→ 规范 schema。聚宽用户的通用取数入口。

    · 凭证走环境变量 JQ_USER/JQ_PASS(只进用户终端,不落文件);
    · 本地缓存 parquet(按 标的+起止 哈希),聚宽额度只花一次,之后离线;
    · adv 直接用聚宽成交额 money 的滚动均值(比 volume×close 估算更准);
    · skip_paused=True → 停牌日剔除(参差面板,沙箱/五柱均支持)。
    """
    import hashlib, json, os
    securities = [securities] if isinstance(securities, str) else list(securities)
    cache_dir = JQ_CACHE_DIR
    key = hashlib.md5(json.dumps([sorted(securities), str(start), str(end)],
                                 ensure_ascii=False).encode()).hexdigest()[:12]
    cache = cache_dir / f"jq_prices_{key}.parquet"
    if cache.exists() and not refresh:
        return pl.read_parquet(cache)
    user, pwd = os.environ.get("JQ_USER"), os.environ.get("JQ_PASS")
    if not user or not pwd:
        raise RuntimeError("首次取数需聚宽凭证(之后走本地缓存):\n"
                           "  JQ_USER=手机号 JQ_PASS=密码 python3 你的脚本.py")
    import jqdatasdk as jq
    jq.auth(str(user), str(pwd))
    pdf = jq.get_price(securities, start_date=str(start), end_date=str(end),
                       frequency="daily", fields=["close", "money"],
                       skip_paused=True, panel=False)
    df = (pl.from_pandas(pdf.reset_index() if "code" not in pdf.columns else pdf)
          .rename({"time": "trading_date", "code": "symbol"})
          .select([pl.col("trading_date").cast(pl.Date), pl.col("symbol").cast(pl.Utf8),
                   pl.col("close").cast(pl.Float64), pl.col("money").cast(pl.Float64)]))
    df = validate(df.drop_nulls(subset=["close"]))
    df = (df.with_columns(pl.col("money").rolling_mean(adv_window, min_samples=1)
                          .over("symbol").alias("adv")).drop("money"))
    cache_dir.mkdir(exist_ok=True)
    df.write_parquet(cache)
    return df


def from_akshare(symbols=None, start=None, end=None) -> pl.DataFrame:
    """用 akshare 拉股指期货主力连续日线 → 规范 schema。需 `pip install akshare` 且联网。
    注意:不同 akshare 版本接口名可能变,落地时按当前版本核对。"""
    try:
        import akshare as ak
    except ImportError as e:
        raise ImportError("需要 akshare:pip install akshare(可选依赖,仅真实数据接入用)") from e
    symbols = symbols or CN_INDEX_FUTURES
    frames = []
    for sym in symbols:
        raw = ak.futures_main_sina(symbol=f"{sym}0")          # 主力连续
        d = pl.from_pandas(raw).rename({"日期": "trading_date", "收盘价": "close",
                                        "开盘价": "open", "最高价": "high",
                                        "最低价": "low", "成交量": "volume"})
        d = d.with_columns([pl.lit(sym).alias("symbol")] +
                           [pl.col(c).cast(pl.Float64) for c in ("open", "high", "low")])
        frames.append(d.select(["trading_date", "symbol", "close", "open", "high", "low", "volume"]))
    df = pl.concat(frames)
    if start: df = df.filter(pl.col("trading_date") >= pl.lit(start).cast(pl.Date))
    if end:   df = df.filter(pl.col("trading_date") <= pl.lit(end).cast(pl.Date))
    return ensure_adv(validate(df))


def from_tushare(ts_codes, start, end, token=None) -> pl.DataFrame:
    """用 tushare 拉期货日线 → 规范 schema。需 `pip install tushare` 与 token。"""
    try:
        import tushare as ts
    except ImportError as e:
        raise ImportError("需要 tushare:pip install tushare(可选依赖)") from e
    pro = ts.pro_api(token)
    frames = []
    for code in ts_codes:
        raw = pro.fut_daily(ts_code=code, start_date=start, end_date=end)
        d = pl.from_pandas(raw).rename({"trade_date": "trading_date", "vol": "volume"})
        d = d.with_columns([pl.col("trading_date").str.to_date("%Y%m%d"),
                            pl.lit(code).alias("symbol")])
        frames.append(d.select(["trading_date", "symbol", "close", "volume"]))
    return ensure_adv(validate(pl.concat(frames)))
