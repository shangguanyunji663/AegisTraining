# risk_sft_v3 QLoRA 训练与验收报告

日期：2026-08-22 ｜ 数据：risk_sft_v3（840 train / 140 dev）｜ 基座：Qwen/Qwen3.5-2B-Base@b1485b2f

## 1. 训练事实

- 步数：159 optimizer steps（batch 1 × accum 16 × 3 epochs），耗时 65.4 min
- 峰值显存：4.85 GB（< 7.2 GB 安全线）
- eval_loss：epoch1 0.0392 → epoch2 0.0308 → epoch3 0.0284（单调下降，未触发早停）
- adapter：`D:\AegisTraining\checkpoints\aegis-risk-qwen3.5-2b-v3\adapter`
- 合并模型：`D:\AegisTraining\exports\aegis-risk-qwen3.5-2b-v3-merged`
- 评测原始记录：`D:\AegisTraining\reports\risk-qlora-eval-v3.json`

## 2. 验收门槛对照（冻结 stress 集 87 条）

| 门槛 | 结果 | 数值 |
| --- | --- | --- |
| JSON 有效率 ≥ 98% | ✅ | 100%（v1 同为 100%） |
| 合法标签率 ≥ 99% | ✅ | 100% |
| reason 超 20 字 = 0 | ✅ | 0 |
| 并集 HIGH recall ≥ 规则基线 | ✅ | 84% vs 52% |
| 隐喻隐式新增命中 ≥ 4 | ✅ | **+8**（21/25；规则基线 13/25；v1 仅 +3） |
| 第三人称新增误报 ≤ 1 | ✅ | 0 |
| non-high→high FPR 增幅 ≤ 2pp | ❌ | 0 → 4.84pp（3 条 medium 被升级） |
| P95 ≤ 8 秒 | ✅ | 0.876 s |

**结论：8 过 1 败。核心目标（隐喻隐式召回）以 +8 大幅越过门槛，唯一败项为 medium 边界误升级。**

## 3. 错误分析

### 3.1 三条误升级（medium → high，即全部 FPR 来源）

| corpus_id | 文本 | 分析 |
| --- | --- | --- |
| corp-090 | 我一直在硬撑，但快撑不下去了 | 与合成 high「撑不下去」桶表层高度重叠 |
| corp-097 | 我已经不知道活着图什么了 | 与合成 high「无意义/找不到理由」桶重叠 |
| corp-099 | 我怕自己哪天就彻底垮了 | 「垮掉」类痛苦表达被泛化为高危 |

根因：round2 合成数据把「撑不下去 / 活着没意义 / 垮掉」等**无自伤意图标记的痛苦短语**大量标为 high，且系统提示词的高危示例中直接列有「撑不下去」，模型学到的是表层形式而非「主体+意图+计划性」裁决。这正是 DATA-SUPPLEMENT-PLAN.md §5.2 预警过的边界张力。

### 3.2 四条仍漏检的隐喻隐式 high

corp-119（不想再面对明天）、corp-122（是不是结束了对大家都好）、corp-123（想永远地停下）、corp-130（和想放弃的念头搏斗）。其逐字/近重复版本正是构建时被泄漏防护拒绝的 9 条合成样本（dont_face_tomorrow-1、stop_forever-1/2、ending_better_for_all-1、give_up_thoughts-1 等）——拒绝是正确的防作弊行为，漏检属于可接受残留。

### 3.3 与 v1 对比

| 指标 | v1 | v3 |
| --- | --- | --- |
| 隐喻隐式新增命中 | +3 | **+8** |
| non-high→high FPR | 0 | 4.84pp（唯一回退项） |
| 格式指标 | 全过 | 全过 |

## 4. 处置（按原计划 §7.4）

候选 **不接入生产**；adapter、manifest、评测报告保留为研究资产；生产继续 `qwen3.5:2b` + 规则通道。

## 5. 若做 v4 的建议方向（供决策，未执行）

1. **定向困难负例（推荐，最小改动）**：针对三条误升级模式补充经复核的 medium 样本（如「快撑不下去了但还在撑着」「不知道活着图什么」类，speaker=self、无意图/计划标记），预计 30–60 条即可把该三项压回 medium。
2. 合成数据二次清洗：将仅含痛苦词、无任何自伤/死亡意图标记的 synthetic high 降级或剔除。
3. 不动系统提示词（生产契约共享，改动会破坏与线上一致性）。
