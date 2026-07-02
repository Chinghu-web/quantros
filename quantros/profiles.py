"""
QuantROS 品种参数档 (Calibration Profiles) —— 把阈值按品种显式化,而非散落的魔法数。

诚实原则:阈值【不能】调到"让某策略通过"(那是对验证器自身过拟合)。
这里给的是按品种特性可辩护的默认值,且每个值都标注依据;真实部署仍需按自己的
账户成本/资金规模微调。贴着噪声的判决由置信度提示(t 值 / PBO 配置数 / 二项 SE)兜底。
"""

PROFILES = {
    # 中国股指期货(中金所 IF/IH/IC/IM),日频、非平今
    "cn_index_futures": {
        "realistic_bps": 3.0,   # 佣金 ~0.5bp/round(中金所 ~0.23‱/边)+ 滑点 ~2bp;远低于股票的 10bp
        "target_aum": 1e7,      # 散户默认目标资金 1000 万元;按需调
        "k": 0.01,              # 平方根冲击系数,粗略数量级,需按真实成交分布校准
    },
    # 通用兜底(偏保守,接近股票/不流动品种)
    "default": {
        "realistic_bps": 10.0,
        "target_aum": 1e8,
        "k": 0.01,
    },
}

# 这些键可被 profile 提供、并被 diagnose 的显式 opts 覆盖
PROFILE_KEYS = ("realistic_bps", "target_aum", "k")


def get_profile(name_or_dict):
    """按名取档,或直接传 dict 自定义。未知名报错(不静默退默认)。"""
    if isinstance(name_or_dict, dict):
        return dict(name_or_dict)
    if name_or_dict not in PROFILES:
        raise ValueError(f"未知 profile: {name_or_dict};可选: {list(PROFILES)}")
    return dict(PROFILES[name_or_dict])
