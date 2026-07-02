"""作弊策略动物园：每只野兽标注应由哪个探针拦截，或标注为已知盲区。"""
import numpy as np
import polars as pl
from quantros.causality import BaseStrategy


class HonestBasisStrategy(BaseStrategy):
    """诚实：时序 symbol 隔离 + 截面 trading_date 对齐。应放行。"""
    def generate_signals(self, df):
        df = df.with_columns(((pl.col("close")-pl.col("spot_close"))/pl.col("spot_close")).alias("br"))
        df = df.with_columns(pl.col("br").rolling_mean(window_size=5).over("symbol").alias("brm"))
        return df.with_columns((pl.col("brm").rank().over("trading_date")/pl.len().over("trading_date")).alias("weight"))


class EvilShiftStrategy(BaseStrategy):
    """野兽1 时序位移 shift(-1)。预期：尖峰探针拦截。"""
    def generate_signals(self, df):
        df = df.with_columns(pl.col("close").shift(-1).over("symbol").alias("t"))
        return df.with_columns(pl.when(pl.col("t")>pl.col("close")).then(1.0).otherwise(-1.0).alias("weight"))


class EvilGlobalMeanStrategy(BaseStrategy):
    """野兽2 全样本 mean/std 标准化。预期：尖峰探针拦截。"""
    def generate_signals(self, df):
        df = df.with_columns([pl.col("close").mean().alias("gm"), pl.col("close").std().alias("gs")])
        df = df.with_columns(((pl.col("close")-pl.col("gm"))/pl.col("gs")).alias("z"))
        return df.with_columns(pl.when(pl.col("z")>0).then(1.0).otherwise(-1.0).alias("weight"))


class EvilPerSymbolRankStrategy(BaseStrategy):
    """野兽3 品种内全样本排名。预期：尖峰探针失明，重采样探针拦截。"""
    def generate_signals(self, df):
        df = df.with_columns((pl.col("close").rank().over("symbol")/pl.len().over("symbol")).alias("pr"))
        return df.with_columns(pl.when(pl.col("pr")>0.5).then(1.0).otherwise(-1.0).alias("weight"))


class EvilHardcodedConstantStrategy(BaseStrategy):
    """
    野兽4 硬编码未来常数（怀疑是已知盲区）。
    作弊者先在本地用【全量数据】跑优化，得出"最佳阈值=某个数"，
    然后把这个数字【硬编码】进策略。这个常数不从 df 数据通路流过，
    所以任何对 df 的扰动（尖峰/重采样）都碰不到它。
    这里模拟：用全样本中位数算出一个魔法阈值，硬编码成字面量。
    """
    # 假设作弊者离线算出 close 的全样本中位数约为 5000，硬编码进来：
    MAGIC_THRESHOLD = 5000.0
    def generate_signals(self, df):
        # 只用当前 close 和一个硬编码常数比较——表面上完全因果安全！
        return df.with_columns(
            pl.when(pl.col("close") > self.MAGIC_THRESHOLD)
              .then(1.0).otherwise(-1.0).alias("weight"))
