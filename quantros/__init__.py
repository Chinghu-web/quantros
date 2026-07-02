"""QuantROS 策略实盘就绪度体检平台 —— 顶层入口。

本地用户的主路(无需上传,全程本地 import):
    import quantros
    from my_strategy import MyStrategy           # 你自己的向量化策略类
    quantros.diagnose(MyStrategy, data)          # 五柱满血体检

进不来向量化契约的人(别的框架,只能导出持仓)走中档:
    quantros.diagnose_outputs(positions_by_config, prices)
"""
from quantros.report import health_report
from quantros.multiconfig import diagnose_multiconfig
from quantros.profiles import get_profile
from quantros.universal import evaluate_returns          # 通用层正门(②③,任何策略类型)
from quantros.sandbox import run_sandbox, run_sandbox_grid  # 时点化沙箱(①,前视物理不可能)
from quantros.verdict import final_verdict               # 五门分级判决(证伪器出口)

__version__ = "0.2.1"


def diagnose(strategy_cls, data, *, profile="default", **opts):
    """完整模式:对一个【向量化策略类】跑五柱体检(因果/过拟合/成本/容量/稳健)。

    strategy_cls : 带 `generate_signals(df) -> df(含 'weight' 列)` 的类;
                   有 `param_grid()` 则解锁过拟合柱。全程本地,代码不出门。
    data         : 行情 DataFrame(见 quantros.data 的规范 schema)。
    profile      : 品种参数档名或 dict(见 quantros.profiles);默认 "default"。
                   股指期货用 "cn_index_futures"。
    其余 opts(realistic_bps / target_aum / k / robustness_paths / name)显式覆盖档内默认值。
    """
    cfg = {**get_profile(profile), **opts}      # 显式 opts 优先级高于 profile
    return health_report(strategy_cls, data, **cfg)


def diagnose_outputs(positions_by_config, prices, **opts):
    """中档模式:只需【多配置持仓 + 价格】,无需源码。解锁 PBO + 成本 + 容量。"""
    return diagnose_multiconfig(positions_by_config, prices, **opts)
