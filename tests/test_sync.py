"""云端接缝回归套件 —— 只送判决不送 IP、HTTP 注入、许可离线宽限。"""
import json
import pytest
from quantros.sync import JSONFileSink, HTTPSink, License, _assert_clean


def _report():
    return {"pbo": 0.42, "overfit_risk": True, "cost_fail_configs": ["a", "b"],
            "per_config": {"a": {"breakeven_bps": 9.1, "cost_fail": True}},
            "not_assessed": ["一·因果"]}


def test_assert_clean_blocks_non_metrics():
    """防御断言:夹带数组/对象(疑似代码或原始持仓)必须被拒绝过网。"""
    dirty = _report(); dirty["raw_positions"] = [[1.0, -1.0], [1.0, 1.0]]
    with pytest.raises(ValueError):
        _assert_clean(dirty)


def test_jsonfile_sink_writes_local(tmp_path):
    out = tmp_path / "verdict.json"
    JSONFileSink(out)(_report())
    loaded = json.loads(out.read_text())
    assert loaded["pbo"] == 0.42


def test_http_sink_sends_only_report_with_key():
    """HTTP 接缝:用注入的 _post 捕获负载,断言只发判决 + 带上 API key。"""
    captured = {}
    def fake_post(url, headers, payload):
        captured.update(url=url, headers=headers, payload=payload); return 200
    rep = _report()
    status = HTTPSink("https://api.quantros.test/v1/reports", "KEY123", _post=fake_post)(rep)
    assert status == 200
    assert captured["payload"] == rep                      # 过网的就是判决本身
    assert captured["headers"]["Authorization"] == "Bearer KEY123"
    assert "raw" not in json.dumps(captured["payload"])    # 没有原始数据字样


def test_license_offline_grace():
    """许可服务异常(断网)→ 离线宽限,不卡住本地计算。"""
    def boom(_): raise ConnectionError("network down")
    assert License("KEY", _verify=boom, grace=True).valid() is True
    assert License("", _verify=boom).valid() is False       # 无 key 直接无效


def test_license_online_verify():
    assert License("GOOD", _verify=lambda k: k == "GOOD").valid() is True
    assert License("BAD", _verify=lambda k: k == "GOOD").valid() is False
