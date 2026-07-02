"""现场演示：python -m quantros.demo"""
from quantros.causality import CausalityTester, generate_mock_futures_data
from quantros.zoo import (HonestBasisStrategy, EvilShiftStrategy, EvilGlobalMeanStrategy,
                          EvilPerSymbolRankStrategy, EvilHardcodedConstantStrategy)
from quantros.generalization import generalization_probe, make_rescaled_market

if __name__ == "__main__":
    df = generate_mock_futures_data()
    t = CausalityTester()
    print("="*60)
    for name, s in [("诚实基差", HonestBasisStrategy()),
                    ("野兽1 shift(-1)", EvilShiftStrategy()),
                    ("野兽2 全样本均值", EvilGlobalMeanStrategy()),
                    ("野兽3 品种内排名", EvilPerSymbolRankStrategy()),
                    ("野兽4 硬编码常数", EvilHardcodedConstantStrategy())]:
        print(f">>> {name}")
        t.verify_causality(s, df)
        print("-"*60)
    print(">>> 野兽4 交给探针C（跨市场泛化）：")
    generalization_probe(EvilHardcodedConstantStrategy(), df, make_rescaled_market(df))

    # ── 第二根柱子：过拟合探针 ──────────────────────────
    from quantros.overfitting import (generate_data_with_edge, oos_degradation_probe,
                                       pbo_probe, RobustMomentumStrategy, OverfitRandomSeedStrategy)
    print("=" * 60)
    print(">>> 过拟合探针（数据内埋有真实动量 edge）")
    dfe = generate_data_with_edge()
    for name, cls in [("诚实动量（应放行）", RobustMomentumStrategy),
                      ("随机种子伪装（应被抓）", OverfitRandomSeedStrategy)]:
        print(f">>> {name}")
        oos_degradation_probe(cls, dfe)
        pbo_probe(cls, dfe)
        print("-" * 60)

    # ── 第三根柱子:成本敏感性探针 ──────────────────────
    from quantros.cost import (cost_sensitivity_probe, LowTurnoverMomentumStrategy,
                               ChurnyMomentumStrategy)
    print("=" * 60)
    print(">>> 成本敏感性探针（弱 edge 数据,真实成本 10bp）")
    dfc = generate_data_with_edge(beta=0.05)
    for name, strat in [("低换手动量（应放行）", LowTurnoverMomentumStrategy()),
                        ("高频空转（应被抓）", ChurnyMomentumStrategy())]:
        print(f">>> {name}")
        cost_sensitivity_probe(strat, dfc)
        print("-" * 60)

    # ── 第四根柱子:容量 / 冲击成本探针 ──────────────────
    from quantros.capacity import generate_capacity_data, capacity_probe
    print("=" * 60)
    print(">>> 容量探针（同一动量信号,目标资金 $1亿）")
    for name, adv in [("高流动性品种（应放行）", 5e8), ("低流动性品种（应被抓）", 5e6)]:
        print(f">>> {name}")
        capacity_probe(LowTurnoverMomentumStrategy(), generate_capacity_data(adv_dollars=adv))
        print("-" * 60)

    # ── 第五根柱子:状态稳健性探针(E1 重采样 + E2 压力)──
    from quantros.robustness import (generate_regime_data, bootstrap_robustness_probe,
                                     stress_probe, AlwaysLongStrategy)
    from quantros.overfitting import RobustMomentumStrategy, OverfitRandomSeedStrategy
    print("=" * 60)
    print(">>> 状态稳健性探针(正向 drift + 真动量 edge 数据)")
    dfr = generate_regime_data(drift=0.0012)
    for name, strat in [("稳健动量（应放行）", RobustMomentumStrategy(5)),
                        ("无 edge 随机（E1 抓）", OverfitRandomSeedStrategy(0)),
                        ("永远满仓多（E1 放行/E2 抓）", AlwaysLongStrategy())]:
        print(f">>> {name}")
        bootstrap_robustness_probe(strat, dfr, n_paths=150)
        stress_probe(strat, dfr)
        print("-" * 60)

    # ── 压轴:五柱合一的实盘就绪度体检报告 ──────────────
    from quantros.report import health_report
    from quantros.cost import ChurnyMomentumStrategy
    from quantros.zoo import EvilShiftStrategy
    dfH = generate_regime_data(drift=0.0012, adv_dollars=5e8)   # 高流动性
    dfL = generate_regime_data(drift=0.0012, adv_dollars=5e6)   # 低流动性
    health_report(RobustMomentumStrategy, dfH, name="稳健动量@高流动性")
    health_report(ChurnyMomentumStrategy, dfH, name="高频空转@高流动性")
    health_report(EvilShiftStrategy, dfH, name="shift(-1)前视@高流动性")
    health_report(AlwaysLongStrategy, dfH, name="永远满仓多@高流动性")
