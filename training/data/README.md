# 风险 SFT 数据规范

训练原始 JSONL 每行一个对象：

```json
{
  "sample_id": "hongzhi-suicide-train-001",
  "message": "我最近觉得自己很多余。",
  "risk_level": "medium",
  "reason": "明显痛苦但无自伤意向",
  "source": "hongzhiq-socialcd-3k",
  "source_version": "SupervisedVsLLM-EfficacyEval@78fb4d1",
  "label_method": "source_label|rule_mapped|project_mapped|manual_reviewed|synthetic",
  "speaker_scope": "self|third_party|fictional",
  "review_status": "source_provided|not_reviewed|manual_reviewed",
  "annotator": "weak-risk-mapping-v2",
  "adjudication_note": ""
}
```

## 标签规范

- `high`：说话人自身明确或隐喻的自杀、自伤意向、计划或即时危险。
- `medium`：显著痛苦、绝望或功能损害，但没有足够的自身自伤证据。
- `low`：一般困扰，以及新闻、论文、影视、朋友或虚构情境中的高危词。
- “撑不下去”“不配”“没意义”等不能脱离主体、意图和计划性，仅凭关键词升为 `high`。
- `reason` 最多 20 个中文字符；训练目标必须是单一 JSON 对象。

## 来源、弱标注与可追溯性

当前第一批数据并非都经过人工复核；`review_status` 与 `label_method` 是如实描述来源质量的元数据，不能把自动映射写成“人工批准”。

- HongzhiQ suicide 二分类原标签：`0 → low`、`1 → high`，但会用主体规则修正明显的第三人称语境或自身高危表达。
- SocialCD-3k 与认知歪曲样本：默认 `medium`；明确的说话人自身自伤信号映射为 `high`；他人、新闻或虚构语境映射为 `low`。
- 项目内部数据：只从 `HEAD` 的 `representative_corpus.json` 中选择 `layer=base` 的 63 条合成、人工标注开发样本；不读取工作区未提交版本。
- 数据 manifest 会记录样本哈希、来源、映射方式、train/dev 分布和构建种子。数据、权重和报告均存放在 `D:\AegisTraining\`，不进入 Git。

## 开发集与最终 holdout

- 可选项目开发样本：`eval/fixtures/representative_corpus.json` 中 `layer=base` 的 63 条，按 `expected_risk` 使用。
- 固定最终 holdout：同一文件中 `layer=stress` 的 87 条。准备脚本对该子集执行精确哈希与近重复检测；它绝不进入 train/dev。
- `risk.json`、`routing.json`、`multi_turn_corpus.json`、`safety.json`、`tests/`、RAG fixtures、Harness、probe、风险政策文档和高风险 Skill 均不作为训练语料。
- `stress` holdout 可用于最终风险分数，但并非独立盲测集：其中的合成表达与仓库规则/测试存在共同设计背景。生产结论仍需额外的外部、专家审阅且完全隔离的测试集。

## 数据规模与版本

早期版本是 720 条 train + 120 条 dev，三类标签尽量均衡；这些数字只用于历史复现。当前推荐流程不是直接训练早期数据，而是：

```text
consolidated_risk_v1
  -> prepare_risk_sft_v4.py
  -> risk_sft_v9: 2867 train / 200 dev / 1414 devtest
  -> train_risk_qlora.py
```

`risk_sft_v9` 的最终规模、类别分布、重标摘要、泄漏拒绝和来源以 `data/risk_sft_v9/manifest.json` 为准；数据质量以 manifest 的 `label_method`、`review_status` 和审计记录为准。
