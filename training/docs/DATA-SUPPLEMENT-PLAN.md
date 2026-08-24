# QLoRA 隐式高危数据补充计划（验收冲刺）

> 分支约束：全部操作限定在 `feat/qlora-risk-training`。训练工件置于 `D:\AegisTraining\`（不进 Git）。
> 状态：**计划稿**，待确认后执行。本文档不修改任何训练数据或代码。

## 1. 现状与缺口（基于已核实事实）

| 项 | 数值 | 来源 |
| --- | --- | --- |
| 验收隐喻集大小 | 25 条（`corp-106..130`，`category=suicidal_implicit`，属 `stress` 层 holdout） | `eval/fixtures/representative_corpus.json` |
| 规则基线命中 | 13 / 25（recall 0.52） | `data/eval/risk_dual_path.json`、`eval_risk_qlora.py` |
| QLoRA 现新增命中 | 3 条 → 并集 16 / 25 | `training/README.md` L116 |
| 验收门槛 | 并集在 `suicidal_implicit` 上**新增 ≥ 4 条**（即 ≥ 17 / 25） | `eval_risk_qlora.py` `_acceptance_gate` L196-198 |
| 当前被规则与 QLoRA **双双漏检** | 9 条（12 条规则漏检中 QLoRA 仅补 3） | 同上推导 |
| 项目 `base` 层（已进训练）高风险样本 | 5 条，全部 `suicidal_explicit`（corp-101..105），**隐喻/隐式 0 条** | `representative_corpus.json` base 层 |
| 现有训练 high 桶构成 | 240 high / 720 train；来源 hongzhiq-suicide 413、socialcd 270、cognitive 10、project-base 27 | `D:\AegisTraining\data\risk_sft_v2\manifest.json` |

**核心结论**：当前训练集几乎没有"项目域隐喻式隐式高危"样本——项目自身的隐喻样本全部在 `stress` holdout（依法不可训练），外部源（hongzhi/socialcd/cognitive）的弱标签映射依赖显式关键词（`source_ingest.py` `SELF_HIGH_PATTERN`），隐喻样本易被降级为 medium/low，覆盖薄弱。这正是 QLoRA 在 25 条隐喻上只加 3 条的根因。

**防作弊保证**：验收严格使用冻结的 25 条 holdout；`prepare_risk_sft.py` 内置 `assert_no_final_holdout_leakage`（精确哈希 + 字符 n-gram 近重复阈值 0.82），任何与 25 条近重复的候选将在准备阶段被拒绝。补充数据**必须**通过该门槛，否则视为泄漏。

## 2. 隐喻表达分类框架（取自 25 条 holdout 的真实模式）

以 `tests/test_risk_dual_channel.py` 的 `METAPHOR_CASES_RULES_MISS`（L32-37）为项目权威隐喻词表，归纳 14 个隐喻桶：

| # | 隐喻桶 | 代表表达 | holdout 示例 |
| --- | --- | --- | --- |
| 1 | 停止/ cessation | 结束这一切、让这一切停下来 | corp-106, 117, 123 |
| 2 | 消失/ disappearance | 想消失、自己消失 | corp-108, 112, 128 |
| 3 | 离世/ departure | 离开这个世界、告别、和这个世界告别 | corp-110, 126, 122 |
| 4 | 否定存在/ non-existence | 没出生过、从没出生过 | corp-111 |
| 5 | 解脱/ relief | 解脱、死了是解脱 | corp-114, 118 |
| 6 | 无法继续/ can't-go-on | 撑不下去、不想再继续、不再继续 | corp-115, 130 |
| 7 | 失去理由/ reason-loss | 没有活下去的理由、活下去的力气 | corp-116, 120 |
| 8 | 长眠/ sleep-death | 永远睡过去、不再醒来、睡过去不再醒来 | corp-121, 129, 108 |
| 9 | 回避未来/ future-avoidance | 不想再面对明天、不想面对明天 | corp-119 |
| 10 | 缺席/ absence | 不在了、要是我不在了就好了 | corp-128 |
| 11 | 放弃/ giving-up | 放弃的念头、和想放弃的念头搏斗 | corp-130 |
| 12 | 委婉 euphemism | 做傻事、怕自己做傻事 | corp-124 |
| 13 | 无价值/ redundancy | 活着多余、没有任何人需要我 | corp-127 |
| 14 | 习语 idiom | 一了百了、不如死了算了 | corp-109, 125 |

补充目标是让训练 high 桶在每个桶上都有**多样表层形式**覆盖，从而泛化到 holdout 中尚未命中的 9 条。

## 3. 数据来源

### 来源 1：项目内未被当前训练覆盖的隐喻样本（预计 30–50 条）
- **1a 项目隐喻词表落地**：基于 `tests/test_risk_dual_channel.py` 的隐喻词表，生成项目域（校园心理）隐喻式隐式高危 utterance，作为带 `manual_reviewed` 标签的高风险样本。
- **1b 项目文档中的用户语态隐喻**：从 `knowledge/`（crisis.md、low-mood-depression-support.md、grief-and-loss.md、learned-helplessness.md）、`skills/`（high_risk_safety_plan、supportive_response_baseline）的示例对话中抽取**用户语态**隐喻表达（仅取用户原话，不取系统回复），经人工复核后标注为 high。
- **1c 其他已提交 fixture**：`multi_turn_corpus.json`、`safety.json` 等中存在的隐喻 utterance（若未被排除列表覆盖），按需抽取。
- 落点：`training/data/supplement_project.jsonl`（schema 同 `training/data/README.md`）。
- ⚠️ 注意：现 `training/README.md` 将 knowledge/skills/safety 等排除出训练。本来源需要对此做**有文档记录的例外**，且仅取"用户原话隐喻"，不引入安全模板，避免模型学会复述模板。需您确认是否放行此例外。

### 来源 2：外部隐式高危边界样本（预计 90–130 条）
- **2a 扩展已批准仓库**：对 `SupervisedVsLLM-EfficacyEval@78fb4d1` 做更细粒度抽取——除现有 hongzhi suicide / socialcd / cognitive 外，补充其内更多**隐喻/委婉/暗示**类正样本，并放宽弱映射使其隐喻样本保留为 high（而非被 `SELF_HIGH_PATTERN` 漏掉降级）。
- **2b 新增公开语料（个人研究、许可证已豁免）**：引入更多中文心理困扰/自杀意念隐喻语料（如 CLPsych/CAMS 风格或公开社交媒体隐喻数据集），仅取隐喻/隐式高危子集，弱映射到 high。
- **2c 困难负样本（防止误报）**：补充 ~20–30 条"隐喻相邻但非高危"的 low/medium 干扰（如第三人称"朋友说想消失"、良性"好累想睡一觉"），以满足 `third_person_new_high_fpr ≤ 1`、`non_high_to_high_fpr ≤ 规则+2pp` 两道精度门槛。
- 落点：`D:\AegisTraining\data\external\supplement\`。

## 4. 补充数量预估

| 类别 | 桶覆盖 | 每条桶表层形式 | 预估新增 | 标签 |
| --- | --- | --- | --- | --- |
| 隐喻隐式高危（来源1+2a+2b） | 14 桶全覆盖 | 8–12 种 | **120–150** | high（manual_reviewed / rule_mapped） |
| 困难负样本（来源2c） | — | — | **20–30** | low / medium |
| **合计** | | | **~140–180 候选** | |

规模适配：建议 `train-size` 由 720 增至 **840**（high/medium/low 各 280），`dev-size` 由 120 增至 **140**；或直接在当前 720 内用新隐喻 high 置换同质非隐喻 high。**推荐新增"隐喻 high 保底配额"**（如 train 60 / dev 15 强制保留），使覆盖确定性而非依赖采样哈希顺序——需在 `prepare_risk_sft.py` 增加 `--metaphor-reserve` 之类的可控改动（小改动，可评审）。

## 5. 筛选标准（硬门槛）

入库前每条候选必须通过：
1. **泄漏防护**：`assert_no_final_holdout_leakage`（精确哈希 + 字符 n-gram ≥ 0.82 拒入），与 25 条 holdout 任何近重复一律剔除。
2. **标签规范**：`high` = 说话人自身明确或隐喻的自杀/自伤意向；"撑不下去/不配/没意义"等脱离主体、意图、计划性不得仅凭关键词升 high（`training/data/README.md` L21-27）。
3. **主体正确**：第三人称/虚构语境（朋友、新闻、影视、论文）映射为 low，绝不作为自身 high 训练。
4. **理由约束**：`reason` ≤ 20 中文字符，单一 JSON 目标。
5. **可追溯**：`source` / `source_version` / `label_method` / `review_status` / `speaker_scope` 完整填写，manifest 记录哈希与构建种子。
6. **内容边界**：仅含"来访者可能说出的隐喻式困扰原话"用于检测训练；不生成任何操作性自伤步骤/方法细节。

## 6. 重新训练与验收步骤（可复现，全部在 `feat/qlora-risk-training`）

```bash
# 0. 锁定分支与环境
git checkout feat/qlora-risk-training
set HF_HOME=D:\AegisTraining\hf-cache
set AEGIS_TRAIN_ROOT=D:\AegisTraining

# 1. 数据收集 → 落到 D:\AegisTraining\data\external\supplement\ 与 training/data/supplement_project.jsonl
#    （按第 3、5 节标准，经 leakage_guard 预检）

# 2. 扩展 source_ingest / prepare 以纳入补充源（如需 --metaphor-reserve 则小改 prepare_risk_sft.py）

# 3. 重建隔离数据（v3），自动执行 holdout 泄漏防护
python scripts/prepare_risk_sft.py \
  --source-root D:\AegisTraining\data\external\supplement \
  --output-root D:\AegisTraining\data\risk_sft_v3 \
  --train-size 840 --dev-size 140

# 4. CUDA dry-run 后正式训练（沿用 risk_qlora_4060.yaml）
python scripts/train_risk_qlora.py --data-root D:\AegisTraining\data\risk_sft_v3 \
  --snapshot-dir D:\AegisTraining\models\Qwen3.5-2B-Base --dry-run
python scripts/train_risk_qlora.py --data-root D:\AegisTraining\data\risk_sft_v3 \
  --snapshot-dir D:\AegisTraining\models\Qwen3.5-2B-Base

# 5. 合并 adapter → 新研究工件（不替换生产模型/规则通道）
python scripts/merge_risk_qlora.py \
  --snapshot-dir D:\AegisTraining\models\Qwen3.5-2B-Base \
  --adapter-dir D:\AegisTraining\checkpoints\aegis-risk-qwen3.5-2b-v2\adapter \
  --output-dir D:\AegisTraining\exports\aegis-risk-qwen3.5-2b-v2-merged

# 6. 冻结验收（重点看 implicit_high_new_hits 与两道精度门槛）
python scripts/eval_risk_qlora.py \
  --original-model qwen3.5:2b \
  --qlora-model-dir D:\AegisTraining\exports\aegis-risk-qwen3.5-2b-v2-merged
```

**验收门槛（须全部通过，`eval_risk_qlora.py` `_acceptance_gate`）**：
- `implicit_high_new_hits_at_least_4`：并集在 25 条隐喻上新增 ≥ 4 条（核心目标，建议冲 ≥ 18–19 留余量）
- `third_person_new_high_false_positives_at_most_1`：第三人称/虚构新增 high 误报 ≤ 1
- `non_high_to_high_fpr_increase_at_most_2pp`：非 high→high 误报率 ≤ 规则基线 + 2pp
- `fused_high_recall_not_below_rules`：并集 high recall 不低于规则基线
- JSON 有效率 ≥ 98%、合法标签率 ≥ 99%、理由超 20 字比例 = 0、P95 ≤ 8s

## 7. 风险与回滚
- 若隐喻保底配额过大导致精度门槛（third_person / non_high_fpr）回退，则下调隐喻占比、增配困难负样本后重训。
- 任一门槛失败：adapter 仅保留为研究资产，**生产继续使用原始 `qwen3.5:2b` + 规则通道**，不写入生产配置（遵循 `training/README.md` L104-116）。
- 生产接入前仍需额外"外部、专家审阅、完全隔离"盲测集（当前 stress holdout 非独立盲测）。

## 8. 需您确认的事项
1. 是否放行"来源 1b"——将 `knowledge/`、`skills/` 中的用户语态隐喻纳入训练（需对现有排除策略做文档化例外）。
2. 是否接受 `train-size` 增至 840（+隐喻 high 保底配额）这一规模适配。
3. 外部新语料（来源 2b）是否限定在现有已批准仓库扩展，还是允许引入新的公开隐喻语料（您已豁免许可证，仅确认范围）。
