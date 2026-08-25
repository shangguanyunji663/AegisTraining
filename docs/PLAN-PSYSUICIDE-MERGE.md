# [Archived] PsySUICIDE × 现有风险 SFT 数据合并执行计划

> 状态：**历史归档，不再作为待执行计划。** 本文件保留 2026-08-22 的数据分析、筛选清洗方案、冲突处理策略与成本估算，仅用于审计和历史背景。
> 当前推荐流程已经改为 `training/data/consolidated_risk_v1` → `training/scripts/prepare_risk_sft_v4.py` → v9 训练/评测；本文提出的 `load_psysuicide_candidate_pool()` 方案未实施，也不应作为当前开发任务入口。
> 生成时间：2026-08-22；归档说明：2026-08-25

---

## 0. 关键事实速览（已核证）

| 项 | 新数据集（PsySUICIDE） | 已有数据（risk_sft_v2 为主） |
| --- | --- | --- |
| 来源 | EMNLP'24 PsyGUARD 自杀检测语料（MIT） | SupervisedVsLLM-EfficacyEval + 项目 corpus base 层 |
| 文件 | `train/valid/test.json`（JSON 数组） | `train.jsonl` / `dev.jsonl`（ChatML）+ `manifest.json` |
| 记录数 | **14,800**（11,835 / 1,480 / 1,485） | 候选池 4,746；train 720 / dev 120（240·240·240 / 40·40·40） |
| 标签体系 | **11 类细粒度多标签**（向量+名称） | 3 级 `low/medium/high`（JSON  verdict） |
| 标注质量 | 人工标注；`labels` 与 `label` 向量 **0 处不一致** | 多为弱监督映射（SocialCD/认知歪曲） |
| 文本重叠 | 与已有 SFT-user 仅 **1** 条；与 round2 **0** 条 | — |

**位置说明**：你给定的 `...\PsySUICIDE\原始数据集` 子目录不存在；实际数据文件直接位于 `D:\AegisTraining\external-data\supplement\PsySUICIDE\` 下（train/valid/test.json + README.md）。已按实际路径读取。

**当前验收瓶颈**（来自 `reports/risk-qlora-transformers-eval-final.json`）：唯一未过门槛的是
`implicit_high_new_hits_at_least_4 = False`（规则基线命中 13/25，QLoRA 仅新增 3 条，需 ≥4 条）。
其余门槛（fused high recall ≥ 规则基线 0.52、第三人称新增 high 误报 ≤1、JSON 有效率、理由≤20 字）均已通过。

---

## 1. 一致性与相关性评估

### 1.1 字段定义冲突
| 维度 | 新数据集 | 已有数据 | 冲突 |
| --- | --- | --- | --- |
| 主键 | `idx`（int，**跨 split 非全局唯一**：train∩valid 共享 1281 个 idx 值） | 无 `idx`，依赖 `sample_id` 唯一 | 新 idx 不可直接作主键 |
| 文本 | `text` | `messages[user].content` | 语义一致，容器不同 |
| 标签 | `labels`（中文类别名）+ `label`（11 维 0/1 向量） | `risk_level`（low/medium/high） | **粒度不一致**：11 类 vs 3 级 |
| 来源 | `data_source`（social_media/client_machine/synthetic/client_counselor） | `source`（hongzhiq-*/project-*） | 命名体系不同 |
| 元数据 | 无审查/范围标注 | `label_method/review_status/speaker_scope` 等 | 新数据集缺治理字段 |

**结论**：字段级不兼容，不能直接拼接。必须经"转换层"映射到统一 `RiskSample` 契约。

### 1.2 格式冲突
- 新：**JSON 数组 + 扁平记录**（分类任务表示）。
- 已有：**JSONL + ChatML `messages`**（SFT 训练表示，`data_contract.RiskSample.to_sft_record()`）。
- 冲突本质：分类格式 ≠ 指令微调格式。训练管线（`train_risk_qlora.py`）只消费 ChatML。
- 处理：新数据集必须被**转换为 ChatML**（见 §4.4），这是合并的物理前提。

### 1.3 标签语义/内容相关性冲突
新 11 类需映射到 3 级。已逐条核证向量索引（见 §4.1 映射表），无歧义。相关性判断：
- 新数据对 **high / medium 的覆盖显著优于现有弱监督来源**（新 high 1,874、medium 1,998 且主要为人工 social_media 标注）。
- 现有 SocialCD-3k 占候选池 medium 的绝大部分（3,257/3,285），属弱映射；新数据可提供更可信的 medium/high 信号。
- **攻击类（用户/他人攻击行为）不属于自身自伤**——须映射为 medium/low，且不得触发"high 必须 self"硬约束（data_contract 第 133 行）。

### 1.4 分布冲突（相关性最敏感项）
新数据经映射后分布：**high 1,874 / medium 1,998 / low 10,928（73% 为 low）**。
现有 v2 候选池：**high 1,007 / medium 3,285 / low 454（medium 偏多）**。
- 若原样并入，整体将被 low 主导（≈78%），破坏模型校准。
- 现有管线 `stratified_split` **强制 1:1:1 均衡**（train_size//3 per level）。
- 这是训练契约，不是"数据冲突"。见 §2 决策点。

---

## 2. 冲突处理策略（以新数据集为准）

原则落地为四条可执行规则：

1. **字段/格式冲突** → 统一采用管线 ChatML（`RiskSample` 契约）；但标签**源自新数据集自身的 11 类体系**，不沿用既有弱监督映射（新胜出）。
2. **标签语义冲突（粒度）** → 新 11 类作为细粒度判定的权威来源；目标固定为 3 级（受生产推理契约约束）。新数据贡献的样本，其 3 级判定由新标签推导。
3. **内容相关性/标注冲突** → 记录级文本重叠几乎为零（仅 1 条），不存在同文本矛盾标注；若个别样本既有弱映射又有新标签，以新标签为准。
4. **分布/均衡** → 属**训练契约**（固定 1:1:1），不视为可覆盖的"数据冲突"。是否维持 1:1:1、是否扩容 train_size，作为**待确认决策**（见 §7）。默认维持 1:1:1，并通过"扩大候选池让新数据的高/中样本被优先抽取"来兑现"新数据优先"。

新增治理字段（补齐 §1.1 缺口）：
- `source = "psysuicide-{data_source}"`（保留来源细分，便于 manifest 溯源）
- `label_method = "source_label"`（新数据自带人工标注，权威）
- `review_status = "source_provided"`
- `speaker_scope`：含"他人攻击行为"且无自身自伤类 → `third_party`；否则 `self`
- `adjudication_note`：写入原始 11 类名称，便于审计与回溯

---

## 3. 合并后数据结构

**新增目录**（保留 v2 作回归基线，符合项目 A/B 与"不覆盖"惯例）：
`D:\AegisTraining\data\archive\risk_sft_v3\`
- `train.jsonl` — ChatML，均衡（默认 500·500·500，见 §7）
- `dev.jsonl` — ChatML，均衡（默认 40·40·40 或按比例）
- `manifest.json` — 含 schema_version、映射表、各级/各 source 计数、provenance、去重与泄漏检查结果
- `psyuicide_test_frozen.jsonl` — **仅评测、不训练**（PsySUICIDE test 1,485 条，作为自杀检测独立评测集）

每条记录 schema（与 v2 完全一致，来自 `data_contract.RiskSample`）：
```json
{"messages":[
  {"role":"system","content":"<RISK_SYSTEM_PROMPT>"},
  {"role":"user","content":"<text>"},
  {"role":"assistant","content":"{\"risk_level\":\"low|medium|high\",\"reason\":\"≤20字依据\"}"}
]}
```

---

## 4. 执行步骤（批准后才执行）

### 4.1 数据映射（11 类 → 3 级，已核证索引）
| idx | PsySUICIDE 类别 | 3 级 | speaker_scope | 备注 |
| --- | --- | --- | --- | --- |
| 0 | 自杀未遂 | high | self | 明确自身高危 |
| 1 | 自杀准备行为 | high | self | 明确自身高危 |
| 2 | 自杀计划 | high | self | 明确自身高危 |
| 3 | 主动自杀意图 | high | self | 明确自身高危 |
| 4 | 被动自杀意图 | medium | self | 被动意念，无计划 |
| 5 | 自伤行为 | high | self | 自身自伤 |
| 6 | 自伤意图 | high | self | 自身自伤 |
| 7 | 用户攻击行为 | medium | self | 攻击他人，非自身自伤 |
| 8 | 他人攻击行为 | low | third_party | 他人伤害，非自身 |
| 9 | 关于自杀的探索 | medium | self | 探索/好奇，无意图 |
| 10 | 与自杀/自伤/攻击行为无关 | low | self | 无关 |

多标签规则：取最高严重度（high > medium > low）。仅当非无关标签**只有 idx8** 时 `speaker_scope=third_party`，否则 `self`。
`reason`（≤20 字）：取中文类别名；多标签用"/"连接并保证 ≤20 字（如"主动自杀意图/他人攻击行为"=13 字）。

### 4.2 数据拆分（§数据拆分）
- **训练/验证来源**：PsySUICIDE `train` + `valid`（13,315 条）参与候选池；`test`（1,485 条）冻结为评测集，**不进入 train/dev**。
- 现有来源：hongzhiq-suicide / socialcd-3k / cognitive / project-base（v2 同款）并入候选池。
- round2 补充：90 条合成隐式高危 + 从 hongzhiq low 标签抽取的隐式高危（直接针对失败门槛）。
- 最终 `stratified_split` 输出 train/dev（1:1:1）。

### 4.3 数据筛选（§数据筛选）
1. **完整性**：剔除空 `text`（实测 0 条）；`labels` 与 `label` 向量不一致者剔除（实测 0 条，作为质量背书）。
2. **信任分层**（解决 §1.4 低质偏多）：
   - 高信任（全留）：`client_counselor`(3,000) + `social_media`(3,800) — 人工来源。
   - 中信任（按类配额抽取，避免 synthetic/client_machine 主导 low）：`synthetic`(4,000) + `client_machine`(4,000)。
   - low 类绝大多数来自 synthetic/client_machine（3,571+3,525），训练 low 配额有限，自然封顶，不额外降权。
3. **安全过滤**：任何映射为 high 的样本强制 `speaker_scope=self`（data_contract 硬约束）；攻击类永不为 high。
4. **去重/防泄漏**：沿用管线 `dedupe_samples`（同文本保留高风险首条）+ `assert_no_final_holdout_leakage`（stress 87 条冻结，近重复阈值 0.82；候选池内部 0.92）。

### 4.4 数据清洗（§数据清洗）
- NFKC 归一化仅用于哈希/泄漏判定，**不改训练原文**（`normalize_message`）。
- 去除首尾空白、折叠多余空白。
- 长度：原文最长 1,286 字符 < 契约上限 1,500，无需截断（保留）。
- `reason` 强制 ≤20 中文字；超限截断并告警。

### 4.5 简单数据处理（§数据处理）
- 新增 `source_ingest.load_psysuicide_candidate_pool(root)`：读取三 split → 应用 §4.1 映射 → 产出 `RiskSample` 列表（`sample_id="psysuicide-{split}-{idx}"`，因 idx 跨 split 不唯一，必须带 split 前缀）。
- 在 `prepare_risk_sft.py` 增加 `--psysuicide-root` 参数，将新来源并入候选池；或新建 `prepare_risk_sft_v3.py` 复用既有分层/去重/泄漏逻辑。
- 经 `to_sft_record()` 生成 ChatML，写 `train.jsonl`/`dev.jsonl`，并产出含映射表与计数的 `manifest.json`。

---

## 5. 训练成本与耗时估算

**环境**（已确认）：RTX 4060 Laptop 8GB；隔离环境 `D:\AegisTraining\envs\qlora-qwen35`；
基座 `Qwen/Qwen3.5-2B-Base`（固定 revision `b1485b2`，2B 参数）；4-bit NF4 QLoRA；
配置 `cutoff_len=512, batch=1, grad_accum=16, epochs=3, lr=1e-4`。

**单样本 token 估算**：system ~150 + user(均值 30 字 ~45) + assistant ~25 ≈ **220 token/条**。
**步数公式**：steps ≈ `3 × train_size / 16`。吞吐按 RTX 4060 4-bit 实测区间 **2–4 秒/步**估算（dry-run 后精确化）。

| 方案 | train_size（高/中/低） | 候选池取用 | 步数 | 训练算力耗时 | 总耗时（含加载/3 轮评估/落盘） |
| --- | --- | --- | --- | --- | --- |
| A（同 v2） | 720（240·240·240） | 仅各 240 | ~135 | ~5–9 min | ~15–30 min |
| **B（推荐）** | **1,500（500·500·500）** | 各 500，优先新数据 human 标注 | ~281 | ~9–19 min | ~25–45 min |
| C（最大多样性） | 3,000（1000·1000·1000） | 各 1000 | ~562 | ~19–37 min | ~40–70 min |

**存储开销**：
- 新增数据：`psyuicide_*` SFT JSONL + manifest ≈ **15–30 MB**（极小）。
- 模型工件（多数已存在）：基站快照 ~4–5 GB（已缓存）、adapter（新）~数十 MB、**合并快照（若 merge）~4–5 GB**。
- 净新增磁盘：若执行 merge，约 **5–6 GB**（主要为合并模型）；不 merge 仅存 adapter 则 <100 MB。

**资源消耗**：GPU 累计约 **0.5–1.2 GPU·小时**；CPU/RAM 占用低；电力可忽略。
**关键前提**：先跑 `--dry-run`（管线已支持）实测 steps/sec，再据此固化耗时，避免 2–4 秒/步的区间偏差。

---

## 6. 后续训练准备事项（批准并执行合并后）

1. **接入管线**：实现 `load_psysuicide_candidate_pool` 并接入 prepare 脚本；维持 `leakage_guard` 冻结 stress 87 条（阈值 0.82/0.92）。
2. **基座闸门**：`base_model_gate` 校验 Qwen3.5-2B-Base 快照结构与 LoRA target 资格。
3. **Dry-run**：CUDA 构造 LoRA 后 dry-run，实测吞吐并回填 §5 耗时。
4. **训练**：`train_risk_qlora.py --data-root .../risk_sft_v3 --snapshot-dir .../Qwen3.5-2B-Base`。
5. **合并**：`merge_risk_qlora.py` 输出 safetensors 快照（仅研究工件）。
6. **验收**（冻结集，门槛来自 config）：
   - `implicit_high_new_hits ≥ 4`（**当前唯一失败项，核心目标**）
   - 第三人称新增 high 误报 ≤ 1；fused high recall ≥ 规则基线 0.52
   - JSON 有效率 ≥ 98%；合法标签率 ≥ 99%；理由 >20 字比例 = 0
   - P95 ≤ 8s；ordered/autonomous/langgraph 三运行时无回归
7. **生产接入**：维持 `RISK_LLM_CHANNEL_ENABLED` 双通道策略（规则永久执行，QLoRA 只能升级风险）；**验收未过前绝不替换 `qwen3.5:2b`**。
8. **文档**：按 `docs/records/` 命名规范记录本轮（建议 `ROUND-12-PSYSUICIDE-MERGE.md`），含全轮次链接。

---

## 7. 风险与待确认决策（需你拍板）

1. **类别均衡是否维持 1:1:1**（§1.4/§2 规则 4）：默认维持，并通过扩容让新数据优先；若你要求"以新数据集分布为准"（即允许 ~73% low），需明确——但会显著拉低 high/medium 密度、不利于失败门槛。
2. **train_size 取值**：推荐 **B（1,500）**；若追求最大多样性选 C（3,000）；若仅做最小验证选 A（720）。
3. **是否纳入 round2 合成隐式高危（90 条）**：强烈建议纳入——它直接针对 `implicit_high_new_hits≥4` 失败项，与 PsySUICIDE 互补（PsySUICIDE 偏显式高危，round2 补隐式隐喻）。
4. **v3 命名/基线**：默认新建 `risk_sft_v3` 保留 v2 作 A/B 基线；若你要求就地更新 v2，请明示（不推荐）。
5. **信任分层强度**：默认"human 全留 + machine 按配额"，是否进一步对 synthetic/client_machine 做质量抽样（如困惑度/毒性过滤）待确认。
6. **标签索引依据**：§4.1 索引映射已由数据统计双重验证（单标签记录 + 多标签向量）一致；正式转换前仍建议对照 PsyGUARD 论文 label legend 做最终签字确认。

---

## 8. 审批后我将执行的动作（当前未执行）

> 仅在你在回复中批准，并确认 §7 各项决策后，才依次执行：

1. 在 `training/src/aegis_training/source_ingest.py` 实现 `load_psysuicide_candidate_pool()`；
2. 新建/改造 prepare 脚本，产出 `risk_sft_v3/{train,dev}.jsonl` + `manifest.json` + `psyuicide_test_frozen.jsonl`；
3. 运行 `base_model_gate` → `--dry-run`（测吞吐）；
4. 执行训练 → 合并 adapter → 在冻结集跑验收；
5. 输出验收报告；**未过门槛则止步于研究工件，不接入生产**。
