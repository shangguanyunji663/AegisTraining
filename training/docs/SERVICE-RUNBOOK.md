# QLoRA 风险推理服务运行手册

本手册针对 `training/scripts/serve_risk_qlora.py`。该脚本是**受控隔离环境的模型端点**，不是带认证、TLS、网关、限流和审计的完整生产服务。默认用途是本机或隔离网段 smoke test；生产部署必须放在受审批的 HTTPS 反向代理/内部服务之后，并保留主项目规则回退。

## 1. 服务边界

```text
生产 Aegis FastAPI
  -> 受保护 HTTPS endpoint
  -> 反向代理/认证/限流/日志脱敏
  -> serve_risk_qlora.py
  -> merged Qwen3.5-2B safetensors
```

模型服务只回答风险 JSON：

- `low`：一般困扰或他人/虚构语境高危词；
- `medium`：显著痛苦、绝望或功能受损但无足够自身自伤证据；
- `high`：说话人自身自伤/自杀意念、计划或即时危险。

服务不负责：回复生成、危机干预、报告审批、工具执行、RAG、用户认证和最终风险融合。

## 2. 启动前检查

确认：

- merged 目录来自通过 gate 的固定基座和指定 adapter；
- `config.json`、tokenizer 文件和所有 safetensors shard 齐全；
- merged manifest、模型 SHA-256、训练报告和 release 状态已保存；
- 使用隔离 Python 环境，不使用生产项目的 Python 环境；
- GPU 驱动、CUDA、Transformers 和模型 dtype 已按验收环境验证；
- 端口没有被无关程序占用；
- `--model-dir` 位于 `AEGIS_TRAINING_ROOT` 下，脚本路径 guard 会拒绝越界目录。

检查文件：

```bat
D:\AegisTraining\envs\qlora-qwen35\Scripts\python.exe ^
  D:\AegisTraining\training\src\aegis_training\base_model_gate.py ^
  --snapshot-dir "D:\AegisTraining\models\Qwen3.5-2B-Base"

dir "D:\AegisTraining\exports\aegis-risk-qwen3.5-2b-v9-merged"
```

## 3. 本机 smoke test

```bat
set AEGIS_TRAINING_ROOT=D:\AegisTraining
set AEGIS_QLORA_MODEL_DIR=D:\AegisTraining\exports\aegis-risk-qwen3.5-2b-v9-merged

D:\AegisTraining\envs\qlora-qwen35\Scripts\python.exe ^
  D:\AegisTraining\training\scripts\serve_risk_qlora.py ^
  --model-dir "%AEGIS_QLORA_MODEL_DIR%" ^
  --host 127.0.0.1 --port 8301
```

启动时会加载完整 merged 模型；首次加载时间和显存占用不等于单次请求延迟。默认 BF16 与冻结验收口径一致；`--load-4bit` 只在显存紧张时使用，启用后必须重新跑冻结评测。

健康检查：

```bat
curl.exe --fail-with-body http://127.0.0.1:8301/health
```

预期：

```json
{"status":"ok","calls":0}
```

## 4. HTTP 契约

### 4.1 `GET /health`

用途：进程和模型已经加载完成后的简单存活检查。

当前响应字段：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `status` | string | 当前为 `ok` |
| `calls` | integer | 进程启动以来成功进入 runner 的累计调用数 |

当前没有 `/ready`、版本、权重 hash、prompt version 和 GPU 状态字段；生产包装层应补充这些信息，不能把 `/health=200` 当作完整 readiness 证明。

### 4.2 `POST /assess`

请求：

```json
{
  "message": "我最近考试压力很大，晚上睡不着"
}
```

成功且模型解析出合法风险 JSON：

```json
{
  "risk_level": "medium",
  "reason": "睡眠困扰和学业压力",
  "latency_ms": 912.0
}
```

模型输出为空或 JSON 解析失败时，服务返回 HTTP 200，但明确表示模型无结果：

```json
{
  "risk_level": null,
  "reason": "",
  "latency_ms": 912.0
}
```

调用方必须把 `risk_level=null` 当作失败并回退规则，不能当成 `low`。推理异常时当前脚本返回 HTTP 500；请求 JSON 无效或 message 为空返回 HTTP 400；未知路径返回 HTTP 404。

请求 smoke test：

```bat
curl.exe -X POST http://127.0.0.1:8301/assess ^
  -H "Content-Type: application/json" ^
  -d "{\"message\":\"我最近考试压力很大，晚上睡不着\"}"
```

高风险 smoke test 只能使用经授权的测试语料，并且不要把真实个人信息写入命令行、终端历史或日志：

```bat
curl.exe -X POST http://127.0.0.1:8301/assess ^
  -H "Content-Type: application/json" ^
  -d "{\"message\":\"我想结束生命\"}"
```

## 5. 生产部署前必须补齐的保护

`serve_risk_qlora.py` 当前是轻量 Python HTTP server，默认没有：

- TLS 和客户端认证；
- Content-Length 上限和请求体大小限制；
- 认证、来源 allowlist、访问令牌或 mTLS；
- 并发 semaphore、最大队列和请求取消；
- `/ready`、模型版本、prompt hash 和权重 hash；
- 结构化指标、日志轮转和进程 supervisor；
- 优雅停止、自动重启、熔断和指数退避；
- 对外错误信息脱敏；
- 端到端审计和请求 ID。

因此生产部署必须：

1. 只绑定受控内网地址，或绑定 loopback 并由反向代理转发；
2. 在代理层完成 TLS、认证、来源限制、请求大小和访问日志脱敏；
3. 在服务层/代理层设置连接超时、读取超时、队列上限和并发上限；8GB GPU 从并发 1 开始压测；
4. 不将用户原文、风险理由、token 或 API key 写入普通日志；
5. 用 Windows Service、计划任务、NSSM 或其他 supervisor 管理启动、重启和优雅停止；
6. 提供独立 `/ready`：模型完成加载、版本和权重 hash 可核对时才返回 ready；
7. 监控请求数、延迟 P50/P95/P99、HTTP 错误、空结果、超时、回退率、GPU 显存和队列长度；
8. 保留上一版 endpoint/model，发生异常时可以立即关闭 QLoRA 开关回到规则通道。

不要把 `--host 0.0.0.0` 直接暴露到公网，也不要把这个脚本当成自带认证的生产网关。

## 6. 主项目调用方的安全语义

主项目配置：

```ini
RISK_QLORA_ENABLED=false
RISK_QLORA_URL=https://approved-qlora.example.com
RISK_QLORA_TIMEOUT_SECONDS=8
```

调用方必须：

- 设置总超时、连接超时和读取超时；
- 对 HTTP 400/404/500/503、网络异常、JSON 解析失败和 `risk_level=null` 统一回退规则；
- 规则通道和 QLoRA 通道取并集，QLoRA 只能提升风险；
- 不允许模型直接决定报告审批、外部工具和安全模板；
- 记录 request ID、模型版本、prompt version、回退原因和延迟，但脱敏保存 message；
- 服务不可达时不能阻塞整个聊天链路或无限重试。

主项目会拒绝 localhost、环回、私有和保留地址作为生产 endpoint。若必须做本机验证，使用测试配置和独立 smoke test，不要把本机地址写进生产 `.env`。

## 7. 运维检查表

启动：

- [ ] 模型目录和 manifest hash 已核对；
- [ ] base gate 通过；
- [ ] 进程使用隔离 Python；
- [ ] host/port 仅绑定受控接口；
- [ ] `/health` 正常；
- [ ] `/assess` low/medium/high 三类脱敏测试通过；
- [ ] 日志没有 API key、原始个人信息和完整敏感语料。

上线：

- [ ] 反向代理 TLS/认证/限流已验证；
- [ ] readiness、版本和 hash 可观测；
- [ ] P95 在 8 秒预算内；
- [ ] 并发 1/2/4 压测完成，GPU 无 OOM；
- [ ] 回退规则和熔断测试完成；
- [ ] 上一版模型或纯规则回滚已演练；
- [ ] release record、审批人和日期已填写。

下线/回滚：

1. 先关闭 `RISK_QLORA_ENABLED`；
2. 确认主项目恢复规则/原模型通道；
3. 停止或隔离异常模型进程；
4. 保存日志、指标、manifest 和错误样本；
5. 恢复上一版 endpoint/model；
6. 在训练谱系和发布记录中记录原因。
