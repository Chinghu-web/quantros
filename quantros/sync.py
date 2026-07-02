"""
QuantROS 云端接缝 (Sync) —— "本地引擎 + 瘦云端"的客户端侧。

设计铁律:只有【判决+指标】的 dict 会过网,代码/数据/原始持仓永不离开本地。
传输层用依赖注入(_post / _verify),因此无需真实服务器即可测试。

变现钩子,渐进三档:
  JSONFileSink  —— 纯本地落盘,不联网(默认)
  HTTPSink      —— 把判决 POST 到你的云端(报告历史 / web 看板 / 团队共享)
  License       —— API key 在线激活 + 离线宽限,门禁
"""
import json


def _assert_clean(report):
    """防御:只允许标量指标 / 标量列表 / 嵌套标量 dict 过网,挡住任何数组/对象类 IP 泄漏。"""
    def ok(v):
        if isinstance(v, (str, int, float, bool)) or v is None:
            return True
        if isinstance(v, list):
            return all(isinstance(x, (str, int, float, bool)) for x in v)
        if isinstance(v, dict):
            return all(ok(x) for x in v.values())
        return False
    bad = [k for k, v in report.items() if not ok(v)]
    if bad:
        raise ValueError(f"拒绝过网:字段 {bad} 含非指标内容(疑似代码/原始数据),不上传")
    return report


class JSONFileSink:
    """把判决写到本地文件,不联网。最保守的默认。"""
    def __init__(self, path): self.path = path
    def __call__(self, report):
        _assert_clean(report)
        with open(self.path, "w") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        return self.path


class HTTPSink:
    """把判决 POST 到云端。_post 可注入(测试用);默认用标准库 urllib。"""
    def __init__(self, url, api_key, _post=None):
        self.url, self.api_key, self._post = url, api_key, (_post or self._default_post)

    @staticmethod
    def _default_post(url, headers, payload):
        import urllib.request
        req = urllib.request.Request(
            url, data=json.dumps(payload, default=str).encode(),
            headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status

    def __call__(self, report):
        _assert_clean(report)        # 只送判决,绝不送 IP
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        return self._post(self.url, headers, report)


class License:
    """API key 许可:在线校验失败时给离线宽限(网络不可用也能继续本地跑)。
    _verify(api_key) -> bool 可注入;默认实现留给落地时接你的许可服务。"""
    def __init__(self, api_key, _verify=None, grace=True):
        self.api_key, self._verify, self.grace = api_key, _verify, grace

    def valid(self) -> bool:
        if not self.api_key:
            return False
        if self._verify is None:
            return self.grace            # 未接校验服务时,按宽限处理
        try:
            return bool(self._verify(self.api_key))
        except Exception:
            return self.grace            # 网络/服务异常 → 离线宽限,不卡住本地计算
