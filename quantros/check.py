"""
QuantROS 真实数据核对 (Real-data Validation) —— 可重跑的端到端体检。

证明引擎在【真实股指期货】上工作,而非只在 mock 上:
  · 数据接入 & schema 规范
  · 因果柱在真实数据上仍能抓 shift(-1) 泄漏(最关键:防"只在 mock 上灵")
  · 因果柱放行诚实因果策略
  · 五柱全链路在真实数据上跑通(用 cn_index_futures 参数档)

运行:  python -m quantros.check   或   quantros-check(装包后)
联网拉数据;失败会给出明确指引并以非零码退出(可用于 CI 守门)。
"""
import quantros
from quantros.causality import CausalityTester
from quantros.zoo import EvilShiftStrategy
from quantros.overfitting import RobustMomentumStrategy


def main(argv=None):
    print("拉取真实股指期货(akshare)...")
    try:
        from quantros.data import from_akshare
        data = from_akshare()
    except Exception as e:
        print(f"❌ 数据接入失败: {e}")
        print("   请检查网络,或 `pip install akshare`;离线时可改用 quantros.data.load_csv(...)")
        return 1

    checks = []
    cols_ok = {"trading_date", "symbol", "close", "adv"}.issubset(data.columns) and len(data) > 0
    checks.append((f"数据接入 & schema(行数={len(data)},品种={data['symbol'].n_unique()})", cols_ok))

    leak_caught = CausalityTester().verify_causality(EvilShiftStrategy(), data, verbose=False) is False
    checks.append(("因果柱在真实数据上抓到 shift(-1) 泄漏", leak_caught))

    honest_ok = CausalityTester().verify_causality(RobustMomentumStrategy(5), data, verbose=False) is True
    checks.append(("因果柱放行诚实因果动量", honest_ok))

    try:
        quantros.diagnose(RobustMomentumStrategy, data, profile="cn_index_futures",
                          robustness_paths=60, verbose=False)
        checks.append(("五柱诊断全链路跑通(cn_index_futures 档)", True))
    except Exception as e:
        checks.append((f"五柱诊断异常: {e}", False))

    print()
    for name, ok in checks:
        print(f"  {'✅' if ok else '❌'} {name}")
    all_ok = all(ok for _, ok in checks)
    print(f"\n核对结果:{'全部通过 ✅' if all_ok else '存在失败 ❌'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
