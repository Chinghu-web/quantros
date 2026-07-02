"""QuantROS 双探针测谎仪 v4 —— 支持探针隔离开关，用于真正的隔离审计。"""
import datetime
import numpy as np
import polars as pl
from typing import List, Set


def generate_mock_futures_data() -> pl.DataFrame:
    symbols = ["IF", "IH", "IC", "IM"]
    start = datetime.date(2026, 1, 1)
    rows, rng = [], np.random.default_rng(42)
    for sym in symbols:
        spot = 4000.0 if sym == "IF" else (3000.0 if sym == "IH" else 6000.0)
        off = rng.uniform(-40, 15)
        for i in range(100):
            d = start + datetime.timedelta(days=i)
            if d.weekday() >= 5: continue
            spot += rng.normal(0, 15)
            fut = spot + off + rng.normal(0, 5)
            rows.append({"trading_date": d, "symbol": sym, "contract": f"{sym}2606",
                "open": fut-10, "high": fut+15, "low": fut-15, "close": fut, "adj_close": fut,
                "volume": float(rng.integers(10000,50000)),
                "open_interest": float(rng.integers(50000,200000)), "spot_close": spot})
    return pl.DataFrame(rows).sort(["symbol","trading_date"])


class BaseStrategy:
    def generate_signals(self, df): raise NotImplementedError


class CausalityTester:
    PROTECTED = {"trading_date","symbol","contract"}
    VALID_PROBES = {"spike", "resample"}

    def __init__(self, injection_ratios=(0.3,0.5,0.7,0.9), spike=1e9,
                 perturb_seed=42, probes: Set[str] = None):
        self.injection_ratios = list(injection_ratios)
        self.spike = spike
        self.rng = np.random.default_rng(perturb_seed)
        # 关键新增：可以指定只跑哪些探针，用于隔离审计
        self.probes = set(probes) if probes is not None else set(self.VALID_PROBES)
        assert self.probes <= self.VALID_PROBES, f"未知探针: {self.probes - self.VALID_PROBES}"

    def verify_causality(self, strategy, df, verbose=True) -> bool:
        df = df.sort(["symbol","trading_date"])
        cols = [c for c,t in zip(df.columns, df.dtypes) if c not in self.PROTECTED and t.is_numeric()]
        try:
            base = strategy.generate_signals(df)["weight"].to_numpy()
        except Exception as e:
            if verbose: print(f" ❌ 策略运行失败: {e}")
            return False
        valid = ~np.isnan(base)
        if not np.any(valid) or valid.sum()/len(base) < 0.5:
            if verbose: print(" ❌ NaN 检验未过")
            return False
        sym_arr = df["symbol"].to_numpy(); symbols = df["symbol"].unique().to_list()

        if "spike" in self.probes:
            for ratio in self.injection_ratios:
                spy = df.clone()
                for sym in symbols:
                    si = np.where(sym_arr==sym)[0]
                    if len(si)==0: continue
                    inj = int(si[int(len(si)*ratio)])
                    for col in cols:
                        spy = spy.with_columns(pl.when(pl.int_range(pl.len())==inj)
                                .then(pl.lit(self.spike)).otherwise(pl.col(col)).alias(col))
                try: cw = strategy.generate_signals(spy)["weight"].to_numpy()
                except Exception as e:
                    if verbose: print(f" ❌ 尖峰失败: {e}"); 
                    return False
                if not self._cmp(df,base,cw,sym_arr,symbols,ratio,"尖峰探针",verbose): return False

        if "resample" in self.probes:
            rand = self.rng.uniform(0.1,5.0,size=len(df))
            for ratio in self.injection_ratios:
                spy = df.with_columns([pl.int_range(pl.len()).over("symbol").alias("_ri"),
                                       pl.Series("_rm",rand)])
                cutoff = pl.col("_ri") >= (pl.len().over("symbol")*ratio).cast(pl.Int64)
                for col in cols:
                    spy = spy.with_columns(pl.when(cutoff).then(pl.col(col)*pl.col("_rm")).otherwise(pl.col(col)).alias(col))
                spy = spy.drop(["_ri","_rm"])
                try: cw = strategy.generate_signals(spy)["weight"].to_numpy()
                except Exception as e:
                    if verbose: print(f" ❌ 重采样失败: {e}"); 
                    return False
                if not self._cmp(df,base,cw,sym_arr,symbols,ratio,"重采样探针",verbose): return False

        if verbose: print(" ✅ 满足因果律。")
        return True

    def _cmp(self, df, base, cw, sym_arr, symbols, ratio, probe, verbose) -> bool:
        for sym in symbols:
            si = np.where(sym_arr==sym)[0]; pidx = si[:int(len(si)*ratio)]
            bs, cs = base[pidx], cw[pidx]
            if not np.array_equal(np.isnan(bs), np.isnan(cs)):
                if verbose: print(f" ❌ [{probe}] NaN 模式改变 sym={sym} ratio={ratio}")
                return False
            m = (~np.isnan(bs)) & (~np.isnan(cs))
            if not np.any(m): continue
            if not np.allclose(bs[m], cs[m], rtol=1e-10, atol=1e-10):
                bad = np.where(~np.isclose(bs[m], cs[m]))[0]; g = int(pidx[m][bad[0]])
                if verbose: print(f" ❌ [{probe}] 泄漏！sym={sym} ratio={ratio} date={df[g,'trading_date']}")
                return False
        return True


# 示例/作弊策略统一收录于 quantros/zoo.py(此处曾有重复定义,已去重)
