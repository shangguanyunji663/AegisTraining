# Aegis Risk QLoRA v9 Release Record

> 当前状态：`release-candidate`，不是 `production-approved`。
> 该记录把 v9 的验收摘要与发布证据状态分开：八项冻结门槛通过，不代表权重下载、哈希、外部审阅、服务安全和生产审批已经完成。

## 1. 版本身份

| 项目 | 当前记录 |
| --- | --- |
| Model version | `aegis-risk-qwen3.5-2b-v9` |
| Dataset | `risk_sft_v9` |
| Prompt contract | `v2` |
| Base repo | `Qwen/Qwen3.5-2B-Base` |
| Base revision | `b1485b2fa6dfa1287294f269f5fb618e03d52d7c` |
| Training repository | `AegisTraining` |
| Training commit | 待发布时填写实际 commit |
| Main project fixture commit | 待发布时填写实际 `aegis-psych-agent` commit |
| Status | `release-candidate` |
| Production switch | `RISK_QLORA_ENABLED=false`，默认关闭 |
| Rollback | 上一版已批准模型或规则通道，待填写具体版本 |

## 2. 当前验收摘要

来源：`reports/V9-TRAINING-EVAL-SUMMARY.md`。

| 门槛 | 阈值 | 结果 |
| --- | --- | --- |
| JSON 有效率 | ≥ 98% | 100% |
| 合法风险标签率 | ≥ 99% | 100% |
| reason 超 20 字比例 | 0 | 0 |
| `rules ∪ QLoRA` HIGH recall | ≥ 规则基线 0.52 | 0.76 |
| 隐喻隐式新增命中 | ≥ 4 / 25 | +6，13→19 |
| 第三人称新增 high 误报 | ≤ 1 | 0 |
| non-high→high FPR 增幅 | ≤ 2pp | 0 |
| P95 延迟 | ≤ 8s | 0.95s |

辅助结果：stress overall accuracy `0.782`、medium recall `0.882`、third-person accuracy `0.818`；devtest 1414 仅作开发参考。

## 3. 训练和数据证据

| 证据 | 路径/状态 |
| --- | --- |
| 参数配置 | `training/configs/risk_qlora_4060.yaml`；hash 待填写 |
| 数据 manifest | `D:\AegisTraining\data\risk_sft_v9\manifest.json`；hash 待填写 |
| 训练 manifest | `D:\AegisTraining\checkpoints\aegis-risk-qwen3.5-2b-v9\training-manifest.json`；hash 待填写 |
| Adapter | `D:\AegisTraining\checkpoints\aegis-risk-qwen3.5-2b-v9\adapter`；文件 hash 待填写 |
| Merged export | `D:\AegisTraining\exports\aegis-risk-qwen3.5-2b-v9-merged`；文件 hash 待填写 |
| Merge manifest | `aegis-export-manifest.json`；hash 待填写 |
| Stress raw report | 待归档 `risk-qlora-eval-v9.json` 和 raw predictions |
| Devtest report | 待归档 `risk-qlora-devtest-v9.json` |
| 外部数据授权 | 需附授权/许可证/脱敏审查记录 |
| 外部审阅 | 待填写审阅人、日期和结论 |

## 4. 训练事实

- 训练数据：2867 train / 200 dev / 1414 devtest。
- LoRA：rank 8、alpha 16、dropout 0.05、4-bit NF4、double quant、BF16。
- 实际可训练参数：8,409,600 / 1,890,234,688，约 0.4449%。
- 训练：540 steps，约 2h54m，峰值 CUDA 显存约 4.9GB。
- `eval_loss`：0.0818 → 0.0638 → 0.0654；最佳 epoch 2 / step 360。

完整参数和命令见 [`training/README.md`](../training/README.md)，证据记录规范见 [`training/docs/REPRODUCIBILITY.md`](../training/docs/REPRODUCIBILITY.md)。

## 5. 发布前 checklist

- [ ] 填写训练仓库 commit 和主项目 fixture commit。
- [ ] 计算 config、data manifest、base snapshot、adapter、merged shard 的 SHA-256。
- [ ] 归档训练 manifest、merge manifest、冻结评测 JSON、raw predictions 和 devtest JSON。
- [ ] 核对 prompt v2 原文和 hash；确认训练/评测/服务/主项目四处一致。
- [ ] 核对 base model license、外部数据授权、脱敏和保留期限。
- [ ] 完成外部专家/安全审阅，记录审阅人、日期、范围和结论。
- [ ] 按 [`training/docs/SERVICE-RUNBOOK.md`](../training/docs/SERVICE-RUNBOOK.md) 完成 TLS、认证、限流、日志脱敏、readiness、回退和 OOM 检查。
- [ ] 完成并发 1/2/4、P95、GPU 显存和超时压测。
- [ ] 完成 `RISK_QLORA_ENABLED` 关闭回退和上一版本恢复演练。
- [ ] 明确生产审批人、日期、endpoint、回滚版本和变更窗口。
- [ ] 所有条件完成后，才可把状态从 `release-candidate` 改为 `production-approved`。

## 6. 已知限制

- 当前 `/health` 是进程健康检查，不是完整 `/ready`；服务版本、权重 hash 和 prompt version 需要由生产包装层补齐。
- `serve_risk_qlora.py` 本身没有 TLS、认证、请求大小限制、队列、熔断和 supervisor，不能直接暴露公网。
- Ollama `qwen3.5:2b` 是 Q8 GGUF，当前没有证明它与固定官方 safetensors revision 同源；推荐 Transformers 隔离服务。
- stress 87 与规则/测试有共同设计背景，不是临床有效性证明；必须补充外部独立测试和专家审阅。
- 模型只能作为风险辅助通道，不能替代心理咨询、危机干预、人工审批或安全政策。

## 7. 回滚步骤

1. 将主项目 `RISK_QLORA_ENABLED` 设为 `false`。
2. 验证主项目恢复规则/原模型通道，检查高风险模板和报告链路。
3. 停止或隔离异常 QLoRA endpoint。
4. 恢复上一版已批准模型或纯规则通道。
5. 保存事故时间线、服务日志、回退率和相关 raw predictions。
6. 在 `reports/TRAINING-HISTORY-INDEX.md` 和本记录中登记原因和处置结果。
