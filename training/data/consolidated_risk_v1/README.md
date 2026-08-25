# consolidated_risk_v1 — 统一风险识别训练/测试集

> 由 `build_consolidated.py` 自动构建。仅读取 `D:\AegisTraining`，未修改/删除其中任何文件。

## 来源（已纳入）
- **PsySUICIDE** (train+valid, 经 hf-mirror 下载, MIT)：自杀意念细粒度 11 标签 → 映射 high/medium/low
- **metaphor_corpus_v1.jsonl**：从 PsySUICIDE 预筛的隐喻/隐式高危子集（按文本去重并入，优先保留其精选标签）
- **suicide 原始集** (`data/suicide/` 下 jsonl/csv/LSAN)：二分类高/低风险

## 来源（已排除，文档记录）
- 外部仓库重复副本 (`external-data/SupervisedVsLLM-EfficacyEval/*`)：与 data/suicide 同源，靠去重覆盖
- 认知歪曲集 (`cognitive distortion` / `SocialCD-3k`)：12 分类任务，非风险识别
- 心理咨询生成集 (`distill_psychology-10k-r1.json`)：无风险标签，属生成目标
- 已派生历史 SFT 集 (`data/archive/risk_sft_v1/v2/v2_round2/v3`)：避免重复计数

## 处理步骤
1. 统一读取为多源记录；2. 剔除空/过短/纯符号/未知风险样本；
3. 按归一化文本精确去重（跨副本）；4. 剔除与 25 条冻结验收集（corp-106..130）
   近重复 (n-gram Jaccard≥0.82) 的泄漏样本；5. 按 risk_level **分层 90/10** 划分 (seed=42)。

本目录的 consolidated 标签是**候选池阶段标签**，不是 v9 最终训练标签。`prepare_risk_sft_v4.py` 会进一步：移除纯攻击/不属于自身风险的记录、重新裁决 low/medium/high、重写 reason、加入已审核困难负例和 medium 扩量、限制第三人称 low 配额，并再次检查 stress 87 泄漏。审计时应同时查看本 manifest 与最终 `data/risk_sft_v9/manifest.json`，不能只看 consolidated 的标签分布。

## 字段（per-sample）
`id, text, risk_level(high|medium|low), reason, source, source_detail, labels_raw, speaker_scope, metaphor_flag, split`

## 产出文件
- `train.jsonl` / `test.jsonl`：统一 per-sample 格式（分析/灵活使用）
- `train_messages.jsonl` / `test_messages.jsonl`：SFT `messages` 中间产物，保留本阶段构建时的旧 system prompt；**当前 v9 不应直接使用此文件训练**。
- `manifest.json`：完整统计与溯源。

当前推荐链路是：

```text
consolidated_risk_v1/train.jsonl
  -> training/scripts/prepare_risk_sft_v4.py
  -> data/risk_sft_v9/train.jsonl + dev.jsonl + test.jsonl
  -> training/scripts/train_risk_qlora.py
```

`prepare_risk_sft_v4.py` 会重新应用当前 prompt contract v2、标签裁决、reason 重写、困难负例/medium 扩量和 stress 泄漏防护。因此，本目录的 `train_messages.jsonl` 只用于审计、历史分析和中间结果检查，不是当前 v9 的权威训练输入。

## 关键统计
- 原始载入：20109 → 无效剔除：2 → 去重后：14553
  → 泄漏剔除：0 → 净样本：14553
- 训练集：13097（high=3371, medium=298, low=9428）
- 测试集：1456（high=375, medium=33, low=1048）

## 重要约定
- 本测试集为**模型开发用 dev-test**，与冻结验收集（corp-106..130，由 `eval_risk_qlora.py` 使用）相互独立。
- 验收前请勿将本集与冻结集混用，以免污染验收结论。
