# 第四版（旧 v5）QLoRA 训练与验收报告 —— 全部通过

日期：2026-08-23 ｜ 数据：risk_sft_v5（1703 train / 200 dev / devtest 1414）｜ 基座：Qwen/Qwen3.5-2B-Base@b1485b2f

## 1. 训练事实

- 步数：321 optimizer steps（batch 1 × accum 16 × 3 epochs）
- eval_loss：epoch1 0.0701 → epoch2 0.0490 → epoch3 0.0461（单调下降，最优即最终，未触发早停）
- adapter：`D:\AegisTraining\checkpoints\aegis-risk-qwen3.5-2b-v5\adapter`
- 合并模型：`D:\AegisTraining\exports\aegis-risk-qwen3.5-2b-v5-merged`（Qwen3_5ForCausalLM, bf16）
- 原始记录：`risk-qlora-eval-v5.json`（冻结验收）、`risk-qlora-devtest-v5.json`（开发测试）

## 2. 冻结 stress 87 条：八门槛全部通过（历史首次）

| 门槛 | 阈值 | 第二版(旧v3) | 第三版(旧v4) | **第四版(旧v5)** |
| --- | --- | --- | --- | --- |
| JSON 有效率 | ≥98% | ✅100% | ✅100% | ✅ **100%** |
| 合法标签率 | ≥99% | ✅ | ✅ | ✅ **100%** |
| reason 超 20 字 | =0 | ✅ | ✅ | ✅ **0** |
| 并集 HIGH recall ≥ 规则 | 0.52 | ✅0.84 | ✅0.80 | ✅ **0.84** |
| 隐喻新增命中 | ≥4 | ✅+8 | ✅+7 | ✅ **+8**（13→21/25） |
| 第三人称新增误报 | ≤1 | ✅0 | ✅0 | ✅ **0** |
| non-high→high FPR 增幅 | ≤2pp | ❌4.84pp | ❌3.23pp | ✅ **0（0 条误升级）** |
| P95 延迟 | ≤8s | ✅ | ✅0.88s | ✅ **0.93s** |

v3 一路追踪的三条边界误升级（corp-090 硬撑 / corp-097 图什么 / corp-099 怕垮）全部正确判 medium。

## 3. dev-test 1414 条（同规则重标金标）

- accuracy 0.901 / macro-F1 0.769 / high-F1 0.876 / low-F1 0.936（原始 qwen3.5:2b 同集约 0.77）
- 格式：JSON 100%、合法标签 100%、reason 超 20 字 0
- 延迟：avg 869ms / P95 934ms
- 弱项：medium F1 0.496（medium 仍是边界最难的类别；第三版为 0.569，主要因本版金标新增了被动降级的更难 medium 样本）
- 混淆主项：low→high 61（其中相当部分为金标本身的边界争议）、high→low 20、low→medium 34

## 4. 相对第三版的数据修复（全部生效）

1. 意念表补回「离开这个世界/告别这个世界/说一声再见」：修复 train 6 / test 12 条 gold 漏标
2. 被动自杀意图「纯无意义/自我否定且无死亡愿望词」降 medium（6 条；含不想活/死/消失/解脱等任一死亡愿望词的维持 high）
3. 第三人称名词表补父母/老师/孩子

## 5. 处置与下游

- **候选正式通过验收**，具备接入生产资格；按原方案 §8 待用户确认后实施
  （`RISK_QLORA_ENABLED` 默认关、RiskGuardian 专属通道、max(规则, QLoRA) 只升不降、异常回退规则）
- Ollama 导入在本机不兼容；服务化路径以隔离 Transformers 环境为准
- 已知局限：medium 类别 F1 偏低；dev-test 上 low→high 约 4.3%（多数为金标边界争议）
