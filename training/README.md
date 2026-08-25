# Aegis 风险 QLoRA 训练主手册

> 本文件是 AegisTraining 的 QLoRA 操作主文档，覆盖环境、基座 gate、数据准备、参数、训练、合并、评测、推理服务、主项目接入、发布、回滚和故障排查。
> 训练仓库与生产应用必须保持隔离：训练代码和 GPU 依赖留在 AegisTraining，生产 FastAPI 进程只通过受保护的 HTTP/JSON 契约调用已验收模型服务。
>
> 相关文档：
> - 数据格式与标签规则：[training/data/README.md](data/README.md)
> - 数据补充计划：[training/docs/DATA-SUPPLEMENT-PLAN.md](docs/DATA-SUPPLEMENT-PLAN.md)
> - 训练谱系：[reports/TRAINING-HISTORY-INDEX.md](../reports/TRAINING-HISTORY-INDEX.md)
> - 当前 v9 验收：[reports/V9-TRAINING-EVAL-SUMMARY.md](../reports/V9-TRAINING-EVAL-SUMMARY.md)
> - 模型发布模板：[docs/MODEL-RELEASES.md](../docs/MODEL-RELEASES.md)
> - v9 发布记录（当前 release-candidate）：[docs/V9-RELEASE-RECORD.md](../docs/V9-RELEASE-RECORD.md)
> - 生产项目接入说明：[aegis-psych-agent/docs/qlora-finetuning.md](https://github.com/shangguanyunji663/aegis-psych-agent/blob/main/docs/qlora-finetuning.md)

## 1. 工程边界和安全原则

AegisTraining 只负责风险识别模型的训练生命周期：

```text
经审查的数据
  -> 数据契约校验与重标
  -> 最终 holdout 泄漏检查
  -> 4-bit NF4 QLoRA 训练
  -> adapter 合并
  -> 冻结集和外部集评测
  -> 版本化发布或回滚
```

生产应用 `aegis-psych-agent` 负责 Agent、RAG、工具治理、审批和回复生成，不在 FastAPI 进程中加载 Transformers、PEFT、bitsandbytes 或训练 adapter。

必须遵守：

- 不把训练依赖追加到生产项目的 `requirements.txt`。
- 不把模型权重、HF cache、checkpoint、merged 权重、GGUF、日志或原始敏感数据提交到 Git。
- 不把 Ollama 的 Q8 GGUF 推理工件当作 Transformers/PEFT 的训练基座。
- 规则风险通道永久保留；QLoRA 只能升级风险，不能降低规则风险。
- 未通过冻结验收、外部审阅和发布清单的模型只能作为研究资产。
- 心理健康和自杀风险数据必须完成授权、脱敏、许可证、用途和公开范围审查。

## 2. 当前推荐版本

| 项目 | 当前值 |
| --- | --- |
| 模型候选 | `aegis-risk-qwen3.5-2b-v9` |
| 数据集 | `risk_sft_v9` |
| 提示词契约 | `v2` |
| 基座 | `Qwen/Qwen3.5-2B-Base` |
| 固定 revision | `b1485b2fa6dfa1287294f269f5fb618e03d52d7c` |
| GPU 对照 | RTX 4060 Laptop，约 8GB 显存 |
| 训练方式 | 纯文本 4-bit NF4 QLoRA |
| 实际训练规模 | 2867 train / 200 dev / 1414 devtest |
| 实测训练 | 540 steps，约 2h54m，峰值显存约 4.9GB |
| 最优 checkpoint | epoch 2，step 360，按 `eval_loss` 选择 |
| 可训练参数 | 8,409,600 / 1,890,234,688，约 0.4449% |
| 冻结最终 holdout | `representative_corpus.json` 的 stress 87 条 |
| v9 结论 | 八项冻结验收门槛全部通过，可作为 release candidate；生产接入仍需显式审批 |

版本编号有历史遗留：磁盘目录可能出现 `v3`、`v5`、`v7`、`v8` 等旧编号，正式谱系以 [TRAINING-HISTORY-INDEX.md](../reports/TRAINING-HISTORY-INDEX.md) 的“第一版～第七版”映射为准。

## 3. 目录和路径约定

```text
AegisTraining/
├── training/
│   ├── configs/risk_qlora_4060.yaml       # 当前 4060/8GB QLoRA 参数
│   ├── scripts/                           # prepare/train/merge/eval/serve 入口
│   ├── src/aegis_training/                 # 数据契约、gate、泄漏、指标、路径工具
│   ├── data/consolidated_risk_v1/          # 经审查的当前数据来源
│   ├── data/authored/                      # 受控人工困难样本
│   └── requirements-qlora.txt              # 独立 GPU 训练依赖
├── models/Qwen3.5-2B-Base/                # 本地官方 safetensors 基座，不进 Git
├── data/risk_sft_v9/                       # 生成的 train/dev/test，不进 Git
├── checkpoints/...                         # PEFT adapter 和 epoch checkpoint，不进 Git
├── exports/...                             # 合并后 safetensors，不进 Git
├── reports/                                # 轻量验收摘要和谱系记录
├── docs/MODEL-RELEASES.md                  # 发布元数据模板
└── envs/qlora-qwen35/                      # 隔离 Python 环境
```

脚本通过 `AEGIS_TRAINING_ROOT` 解析训练根目录，默认是当前 checkout；模型、数据、checkpoint 和导出文件都必须位于训练根目录允许的路径下。路径检查会拒绝越界路径，避免把输出写到未授权目录。

Windows `cmd.exe` 示例：

```bat
set AEGIS_TRAINING_ROOT=D:\AegisTraining
set AEGIS_PROJECT_ROOT=D:\PythonProject\aegis-psych-agent
set AEGIS_PROJECT_CORPUS=%AEGIS_PROJECT_ROOT%\eval\fixtures\representative_corpus.json
set HF_HOME=%AEGIS_TRAINING_ROOT%\hf-cache
```

如果两个仓库不在同一台机器上，应把已审查、已版本化的 fixture 作为明确输入提供给训练流程，不要把本机绝对路径硬编码进数据文件或 manifest。

## 4. 隔离环境

推荐 Python 3.11、与显卡驱动匹配的 CUDA PyTorch，以及独立的 QLoRA 环境。先安装 GPU 版 PyTorch，再安装训练依赖：

```bat
D:\AegisTraining\envs\qlora-qwen35\Scripts\python.exe -m pip install --upgrade pip
D:\AegisTraining\envs\qlora-qwen35\Scripts\python.exe -m pip install -r D:\AegisTraining\training\requirements-qlora.txt
```

当前训练依赖最低约束：

| 包 | 版本要求/用途 |
| --- | --- |
| `transformers` | `>=5.2.0`，Qwen3.5 原生 Transformers 支持 |
| `peft` | `>=0.18.0`，LoRA adapter |
| `bitsandbytes` | `>=0.49.0`，4-bit NF4 与 8-bit optimizer |
| `accelerate` | `>=1.12.0`，设备和训练加速 |
| `datasets` | `>=4.4.0`，JSONL 数据集 |
| `safetensors` | `>=0.6.2`，安全权重读写 |
| `PyYAML` | `>=6.0.2`，读取 YAML 配置 |
| `scikit-learn` | `>=1.6.0`，离线指标辅助 |

PyTorch/CUDA wheel 必须根据本机驱动从官方渠道单独安装；不能因为 `pip install -r` 成功就认为 CUDA 训练可用。训练前检查：

```bat
D:\AegisTraining\envs\qlora-qwen35\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

必须同时确认 `torch.cuda.is_available()` 为 `True`、显存足够，并且 BF16 支持情况与配置一致。若 GPU 不支持 BF16，训练脚本只有在 `fp16_fallback: true` 时才允许自动降为 FP16。

## 5. 基座模型 gate

训练只能使用固定 revision 的官方 Qwen3.5-2B-Base safetensors 快照：

```text
repo:       Qwen/Qwen3.5-2B-Base
revision:   b1485b2fa6dfa1287294f269f5fb618e03d52d7c
model_type: qwen3_5
hidden:     2048
text layers: 24
```

快照至少应包含 `config.json` 和 `model.safetensors.index.json`，并且权重清单非空。执行 gate：

```bat
D:\AegisTraining\envs\qlora-qwen35\Scripts\python.exe ^
  D:\AegisTraining\training\src\aegis_training\base_model_gate.py ^
  --snapshot-dir "D:\AegisTraining\models\Qwen3.5-2B-Base"
```

gate 会检查：

- repo、revision、model type、架构、hidden size、层数和权重清单；
- 是否为可训练 safetensors，而不是 Ollama GGUF；
- 模型是否暴露视觉模块；训练脚本只接受文本 causal-LM 路径；
- 可选的 Ollama 转换 provenance。没有精确转换证明时状态是 `pass_same_family_with_caveat`，不能声称 Ollama 工件与官方 revision 已证明同源。

严格的 `--require-exact-ollama-provenance` 只用于已有完整转换证明的场景；当前本机 `qwen3.5:2b` 没有这样的证明，因此不应强行通过严格模式。

## 6. 数据契约和数据构建

### 6.1 原始数据契约

原始 JSONL 每行至少包含：

```json
{
  "sample_id": "example-001",
  "message": "我最近觉得自己很多余。",
  "risk_level": "medium",
  "reason": "明显痛苦但无自伤意向",
  "source": "curated-source",
  "source_version": "source@revision",
  "label_method": "manual_reviewed",
  "speaker_scope": "self",
  "review_status": "manual_reviewed"
}
```

校验规则：

- `risk_level` 只能是 `low`、`medium`、`high`；
- `reason` 最多 20 个字符；
- 原始 message 最多 1500 个字符；
- `speaker_scope` 只能是 `self`、`third_party`、`fictional`；
- `high` 必须描述说话人自身风险；
- `sample_id` 不得重复；
- `label_method` 和 `review_status` 必须如实记录，自动映射不能伪装成人工审核。

标签原则：

- `high`：说话人自身明确或隐喻的自杀、自伤意向、计划或即时危险；
- `medium`：明显痛苦、绝望或功能损害，但没有足够自身自伤证据；
- `low`：一般困扰，以及新闻、论文、影视、朋友或虚构情境中的高危词；
- “撑不下去”“不配”“没意义”等不能脱离主体、意图和计划性，仅凭关键词升级为 `high`。

### 6.2 当前数据来源和规模

当前主流程使用已审查的 `training/data/consolidated_risk_v1`，通过 `prepare_risk_sft_v4.py` 生成 `data/risk_sft_v9`：

| 输出 | 当前规模 | 用途 |
| --- | ---: | --- |
| `train.jsonl` | 2867 | 训练 |
| `dev.jsonl` | 200 | epoch 评估、early stopping 和最优模型选择 |
| `test.jsonl` | 1414 | devtest 参考评测，不作为最终冻结验收集 |
| 项目 `base` | 63 | 可作为开发候选 |
| 项目 `stress` | 87 | 永久冻结最终 holdout，不能进 train/dev |

默认构建参数：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--train-quota` | `1000 1000 1000` | high / medium / low 目标配额 |
| `--dev-quota` | `70 60 70` | high / medium / low 开发集配额 |
| `--max-text-chars` | `300` | v9 构建时文本长度上限 |
| `--leak-threshold` | `0.82` | 与 stress 的近重复阈值 |
| `--medium-upsample-cap` | `1.35` | medium 扩量上限控制 |
| `--distill-limit` | `200` | 蒸馏来源挖掘上限 |
| 随机种子 | `42` | 确定性哈希抽样 |

实际 train 行数可能高于三类配额之和，因为人工困难负例、成对对照、第三人称负例和审核后的 medium 扩量样本会整组加入训练；最终以 `manifest.json` 为准。

### 6.3 提示词契约 v2

当前 system prompt 必须和训练、评测、推理服务保持一致：

```text
你是校园心理支持系统的风险评估器,只依据当下说话人的**自身意向**判断风险:high(自伤/自杀意念或计划,包括隐喻式表达如想消失/从没出生过/撑不下去/永远睡过去/离开这个世界/结束这一切/不愿再醒来/要是不在了就好了),medium(强烈痛苦、绝望、功能受损但无直接自伤)、low(一般困扰,包括提及他人/虚构情境的高危词,如"新闻里有人轻生/写论文提到自杀/朋友直播自杀"——这些不应升为自身风险)。判定原则:仅评估说话人自身;他人或虚构内容提及自杀/伤害不视为自身 high。只输出一个 JSON 对象,不要输出任何其他文字:{"risk_level": "low|medium|high", "reason": "20字以内依据"}
```

v1→v2 的关键变化是移除“**不配**”“**活着多余**”等宽泛示例，避免与冻结金标中的 medium/high 边界冲突。三处副本必须同步：`training/src/aegis_training/data_contract.py`、生产项目风险提示词和数据构建脚本。

### 6.4 泄漏防护

`prepare_risk_sft_v4.py` 会：

1. 读取项目 fixture 中的 stress 87 条，并确认数量准确；
2. 对候选文本做 NFKC、去空白、转小写规范化；
3. 先用规范化文本哈希拒绝精确复用；
4. 再用字符 3-gram + Jaccard/SequenceMatcher 检测近重复，默认阈值 `0.82`；
5. 将泄漏原因写入 manifest 的 `leakage_rejected`；
6. 对 train 内部和 dev 也做去重，避免验证样本被训练集吸收。

`risk.json`、`routing.json`、`multi_turn_corpus.json`、`safety.json`、RAG fixtures、Harness、probe、政策文档、Skill 内容和测试消息不得作为训练语料。stress 是工程冻结集，不是完全独立的临床盲测集；上线前仍需外部专家审核和独立测试。

### 6.5 可选：重建 consolidated 候选池

当前 v9 主流程默认使用已审查的 `training/data/consolidated_risk_v1`。只有需要重建候选池、审计来源或在受控数据环境中复现实验时，才运行 `build_consolidated.py`：

```bat
set AEGIS_CONSOLIDATED_ROOT=D:\AegisTraining\training\data\consolidated_risk_v1
python D:\AegisTraining\training\data\consolidated_risk_v1\build_consolidated.py
```

重建前必须准备并记录：

- 项目 `representative_corpus.json`，其中冻结输入应保持当前版本；
- PsySUICIDE、suicide 原始数据和 `metaphor_corpus_v1.jsonl` 的来源、许可证、获取时间和 SHA-256；
- `AEGIS_TRAINING_ROOT`、`AEGIS_PROJECT_ROOT`、`AEGIS_PROJECT_CORPUS`；
- 外部数据脱敏、授权和公开范围审查结果。

该脚本只读训练根目录，按归一化文本去重，并对历史 25 条冻结样本做 3-gram Jaccard `0.82` 泄漏过滤。它生成的 `train_messages.jsonl` 使用构建阶段的旧 prompt，是审计中间产物；当前 v9 仍必须经过下一步 `prepare_risk_sft_v4.py` 重新裁决和生成。

### 6.6 构建当前数据

```bat
D:\AegisTraining\envs\qlora-qwen35\Scripts\python.exe ^
  D:\AegisTraining\training\scripts\prepare_risk_sft_v4.py ^
  --consolidated-root "D:\AegisTraining\training\data\consolidated_risk_v1" ^
  --corpus "%AEGIS_PROJECT_CORPUS%" ^
  --output-root "D:\AegisTraining\data\risk_sft_v9"
```

构建后必须检查：

```bat
python -c "import json; p=r'D:\AegisTraining\data\risk_sft_v9\manifest.json'; m=json.load(open(p,encoding='utf-8')); print(m['schema_version']); print(m['train']); print(m['dev']); print(m['devtest']); print('leaks=',len(m['leakage_rejected']))"
```

预期 `schema_version` 为 `risk_sft_v9`，`leakage_rejected` 应被审阅，不能只看命令是否返回 0。

## 7. QLoRA 参数

唯一参数源是 `training/configs/risk_qlora_4060.yaml`。不要把参数散落复制到多个脚本；变更配置后应保存新的 manifest、checkpoint 和验收报告。

### 7.1 基座

```yaml
repo_id: Qwen/Qwen3.5-2B-Base
revision: b1485b2fa6dfa1287294f269f5fb618e03d52d7c
trust_remote_code: false
text_only: true
```

### 7.2 4-bit 量化

| 参数 | 值 | 目的 |
| --- | --- | --- |
| `load_in_4bit` | `true` | 低显存加载基座 |
| `bnb_4bit_quant_type` | `nf4` | NormalFloat4 量化 |
| `bnb_4bit_use_double_quant` | `true` | 二重量化 |
| `bnb_4bit_compute_dtype` | `bfloat16` | 计算精度；不支持时可 fallback 到 fp16 |

### 7.3 LoRA

| 参数 | 值 |
| --- | --- |
| `rank` / `r` | `8` |
| `alpha` / `lora_alpha` | `16` |
| `dropout` | `0.05` |
| `bias` | `none` |
| `task_type` | `CAUSAL_LM` |
| 目标模块 | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`, `in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, `out_proj` |

目标模块不是盲目强制全命中：训练脚本会和实际量化文本模型的线性层后缀取交集；一个都不存在则失败。视觉、图像、视频模块会被排除。

### 7.4 Trainer 参数

| 参数 | 值 |
| --- | --- |
| `cutoff_len` | `512` |
| `per_device_train_batch_size` | `1` |
| `per_device_eval_batch_size` | `1` |
| `gradient_accumulation_steps` | `16` |
| `gradient_checkpointing` | `true` |
| `learning_rate` | `0.0001` |
| `weight_decay` | `0.0` |
| `num_train_epochs` | `3` |
| `warmup_ratio` | `0.05` |
| `lr_scheduler_type` | `cosine` |
| `optim` | `paged_adamw_8bit` |
| `max_grad_norm` | `1.0` |
| `logging_steps` | `10` |
| `eval_strategy` | `epoch` |
| `save_strategy` | `epoch` |
| `save_total_limit` | `2` |
| `seed` | `42` |
| `bf16` | `true` |
| `fp16_fallback` | `true` |
| `early_stopping_patience` | `2` |
| `load_best_model_at_end` | `true` |
| 最优指标 | `eval_loss`，越低越好 |
| `report_to` | `none` |

训练脚本使用 `apply_chat_template`，对 user/system prompt 的 labels 写 `-100`，只对最后 assistant 回复计算 loss；如果 assistant 目标被 `cutoff_len` 截断，会直接报错而不是训练空目标。

## 8. 训练流程

### 8.1 Dry-run

Dry-run 会加载依赖、验证 CUDA、读取配置、执行基座 gate、检查目标层、构造 tokenized train/dev，并输出可训练参数和显存设备信息，但不会调用 `trainer.train()`：

```bat
D:\AegisTraining\envs\qlora-qwen35\Scripts\python.exe ^
  D:\AegisTraining\training\scripts\train_risk_qlora.py ^
  --data-root "D:\AegisTraining\data\risk_sft_v9" ^
  --snapshot-dir "D:\AegisTraining\models\Qwen3.5-2B-Base" ^
  --dry-run
```

dry-run 输出至少应确认：

- CUDA 设备名称；
- 实际 `compute_dtype`；
- 实际匹配的 target modules；
- `train_rows`、`dev_rows`；
- `trainable_params`、`total_params`、`trainable_percent`；
- gate 状态为 `pass_exact` 或明确的 `pass_same_family_with_caveat`。

### 8.2 正式训练

```bat
D:\AegisTraining\envs\qlora-qwen35\Scripts\python.exe ^
  D:\AegisTraining\training\scripts\train_risk_qlora.py ^
  --data-root "D:\AegisTraining\data\risk_sft_v9" ^
  --snapshot-dir "D:\AegisTraining\models\Qwen3.5-2B-Base" ^
  --output-root "D:\AegisTraining\checkpoints\aegis-risk-qwen3.5-2b-v9"
```

必须显式传入 `--output-root`。配置文件中的默认输出目录是通用实验目录，不会自动带上 `v9` 版本号；训练输出目录、adapter 路径和后续 merge 命令必须保持同一个版本标识。

需要改参数时复制配置到一个版本化实验文件并显式传入：

```bat
python training\scripts\train_risk_qlora.py ^
  --config "D:\AegisTraining\training\configs\risk_qlora_4060.yaml" ^
  --data-root "D:\AegisTraining\data\risk_sft_v9" ^
  --snapshot-dir "D:\AegisTraining\models\Qwen3.5-2B-Base" ^
  --output-root "D:\AegisTraining\checkpoints\aegis-risk-qwen3.5-2b-exp01"
```

训练产物结构：

```text
checkpoints/aegis-risk-qwen3.5-2b-v9/
├── adapter/
├── checkpoint-360/
├── checkpoint-540/
└── training-manifest.json
```

v9 实际记录：`eval_loss` 0.0818 → 0.0638 → 0.0654，最佳 epoch 2 / step 360；共 540 steps，约 2h54m，峰值 CUDA 显存 5,072,041,472 bytes（约 4.9GB）。

### 8.3 合并 adapter

合并前 adapter 必须存在 `adapter_config.json` 和 `adapter_model.safetensors`，输出目录必须为空或不存在：

```bat
D:\AegisTraining\envs\qlora-qwen35\Scripts\python.exe ^
  D:\AegisTraining\training\scripts\merge_risk_qlora.py ^
  --snapshot-dir "D:\AegisTraining\models\Qwen3.5-2B-Base" ^
  --adapter-dir "D:\AegisTraining\checkpoints\aegis-risk-qwen3.5-2b-v9\adapter" ^
  --output-dir "D:\AegisTraining\exports\aegis-risk-qwen3.5-2b-v9-merged"
```

合并行为：

- 用 BF16 加载官方基座；
- `PeftModel.from_pretrained` 加载 adapter；
- `merge_and_unload(safe_merge=True)`；
- safe serialization 保存 safetensors，默认最大分片 `2GB`；
- 设置确定性生成：`do_sample=false`、`temperature=null`；
- 写入 `aegis-export-manifest.json`，记录基座 gate、adapter、输出目录、dtype 和分片大小。

### 8.4 冻结集评测

```bat
D:\AegisTraining\envs\qlora-qwen35\Scripts\python.exe ^
  D:\AegisTraining\training\scripts\eval_risk_qlora.py ^
  --original-model qwen3.5:2b ^
  --qlora-model-dir "D:\AegisTraining\exports\aegis-risk-qwen3.5-2b-v9-merged" ^
  --timeout 8 ^
  --max-new-tokens 64 ^
  --output "D:\AegisTraining\reports\risk-qlora-eval-v9.json"
```

评测脚本会：

1. 从项目 fixture 只读取 `layer=stress` 的 87 条；
2. 同时测规则基线、原始模型、QLoRA 原始输出；
3. 计算 `rules ∪ original` 和 `rules ∪ qlora`；
4. 解析 JSON 有效率、合法标签率、reason 长度、accuracy、macro-F1、每类 precision/recall/F1、high recall、non-high→high FPR、medium→high rate、P95；
5. 按 base/stress、implicit high、direct high、third person 分层；
6. 保存 raw predictions，便于追踪单条误判。

`--qlora-model` 用于 Ollama tag，`--qlora-model-dir` 用于合并后的 Transformers safetensors，二者互斥。当前推荐 Transformers 路径；Ollama 的 `qwen3.5:2b` 是 Q8 GGUF，不是训练基座，发布兼容性另行审查。

### 8.5 devtest 参考评测

`eval_risk_devtest.py` 用 1414 条 `risk_sft_v9/test.jsonl` 做开发测试参考，不替代最终 stress 87 验收，也不应把旧脚本默认的 v4 路径当作当前入口：

```bat
D:\AegisTraining\envs\qlora-qwen35\Scripts\python.exe ^
  D:\AegisTraining\training\scripts\eval_risk_devtest.py ^
  --model-dir "D:\AegisTraining\exports\aegis-risk-qwen3.5-2b-v9-merged" ^
  --test-jsonl "D:\AegisTraining\data\risk_sft_v9\test.jsonl" ^
  --output "D:\AegisTraining\reports\risk-qlora-devtest-v9.json" ^
  --max-new-tokens 64
```

该脚本输出 accuracy、macro-F1、每类 F1、non-high→high FPR、medium→high rate、reason/JSON 有效率和 P95，并保留最多 40 条误判样本。devtest 结果用于开发诊断，不能替代冻结 stress 87 的上线 gate。

### 8.6 启动推理服务

```bat
set AEGIS_QLORA_MODEL_DIR=D:\AegisTraining\exports\aegis-risk-qwen3.5-2b-v9-merged
D:\AegisTraining\envs\qlora-qwen35\Scripts\python.exe ^
  D:\AegisTraining\training\scripts\serve_risk_qlora.py ^
  --model-dir "%AEGIS_QLORA_MODEL_DIR%" ^
  --host 127.0.0.1 --port 8301
```

可选 `--load-4bit` 以降低推理显存，但它可能改变边界样本预测；验收采用 BF16，生产切换 4-bit 前必须重新评测。

接口：

```text
GET  /health
POST /assess
```

健康检查示例：

```bat
curl.exe http://127.0.0.1:8301/health
```

请求：

```bat
curl.exe -X POST http://127.0.0.1:8301/assess ^
  -H "Content-Type: application/json" ^
  -d "{\"message\":\"我最近考试压力很大，晚上睡不着\"}"
```

响应为：

```json
{"risk_level":"medium","reason":"睡眠困扰和学业压力","latency_ms":912.0}
```

服务端使用与验收相同的 v2 prompt、`temperature=0`、`max_new_tokens=64` 和宽容 JSON 解析；解析失败返回 `risk_level: null`，由生产调用方回退规则。

## 9. v9 验收门槛和结果

最终门槛以冻结 stress 87 条为准：

| 门槛 | 阈值 | v9 |
| --- | --- | --- |
| JSON 有效率 | ≥ 98% | ✅ 100% |
| 合法风险标签率 | ≥ 99% | ✅ 100% |
| reason 超 20 字比例 | 0 | ✅ 0 |
| `rules ∪ QLoRA` HIGH recall | 不低于规则基线 0.52 | ✅ 0.76 |
| 隐喻隐式高危新增命中（25 条） | ≥ 4 | ✅ +6，13→19 |
| 第三人称/虚构干扰新增 high 误报 | ≤ 1 | ✅ 0 |
| non-high→high FPR 增幅 | ≤ 2pp | ✅ 0 |
| P95 延迟 | ≤ 8s | ✅ 0.95s |

辅助指标：stress overall accuracy `0.782`、medium recall `0.882`、third-person accuracy `0.818`；devtest 1414 的参考结果为 accuracy `0.895`、high-F1 `0.886`、FPR `0.050`。devtest 不是冻结最终上线门槛。

如果任一门槛失败：

1. 不注册生产服务或 Ollama tag；
2. 记录完整 manifest、配置、commit、环境和原始评测报告；
3. 将 adapter 标记为 research-only；
4. 保留上一版已批准模型或规则通道；
5. 先分析分层误判和提示词/数据变化，再决定下一轮训练。

## 10. 主项目接入契约

生产项目的配置：

```ini
RISK_QLORA_ENABLED=false
RISK_QLORA_URL=https://qlora-endpoint.example.invalid
RISK_QLORA_TIMEOUT_SECONDS=8
```

接入规则：

- `RISK_QLORA_ENABLED` 默认 `false`，未明确启用时行为不变；
- 主项目通过 `POST /assess` 调用，不在 FastAPI 进程中加载训练模型；
- 规则评估永久执行，融合采用 `max(规则风险, QLoRA 风险)`；
- QLoRA 不能把规则的 high 降为 medium/low；
- 超时、网络错误、HTTP 错误、JSON 非法或 `risk_level` 缺失都回退规则；
- 主项目 URL 校验拒绝 localhost、环回、私有和保留地址，生产应使用经过审批的公网 HTTPS endpoint；
- 高风险报告、审批、工具调用和安全模板不由模型直接决定，仍由主项目业务链路控制。

当前本机 `127.0.0.1:8301` 服务只用于独立 smoke test。生产接入前必须有进程守护、`/ready`、并发限制、日志脱敏、熔断、监控和回滚版本；详见主项目的 `docs/QLORA-SSE-PRODUCTION-IMPROVEMENTS.md`。

## 11. 发布、校验和回滚

每个 release candidate 必须具备以下元数据：

- 模型名、版本和训练仓库 commit；
- 基座 repo、固定 revision、许可证和下载来源；
- 数据 manifest、数据来源授权、脱敏状态、`label_method` 和 `review_status`；
- system prompt/risk contract 版本；
- adapter 与 merged 文件的 SHA-256；
- 训练配置文件和 `training-manifest.json`；
- 冻结 holdout、外部测试集、延迟和显存结果；
- 生产接入状态、已知限制和回滚版本。

Windows 计算文件哈希：

```powershell
Get-FileHash D:\AegisTraining\exports\aegis-risk-qwen3.5-2b-v9-merged\*.safetensors -Algorithm SHA256
```

发布元数据填写 [docs/MODEL-RELEASES.md](../docs/MODEL-RELEASES.md)。权重应放在受控对象存储、Hugging Face 或经过审查的 Release，不要放回代码仓库。回滚顺序：关闭 `RISK_QLORA_ENABLED` → 恢复上一版已批准 endpoint/model → 验证规则和 API → 保存回滚事件记录。

## 12. 故障排查

| 现象 | 常见原因 | 处理 |
| --- | --- | --- |
| `CUDA PyTorch is required` | CPU 版 PyTorch 或 CUDA 不可用 | 安装匹配驱动的 GPU wheel，先验证 `torch.cuda.is_available()` |
| base gate 缺少 `model.safetensors.index.json` | 使用了 GGUF、非官方快照或不完整下载 | 重新取得固定 revision 的官方 safetensors |
| gate 拒绝 vision modules | 加载了多模态路径 | 使用纯文本 causal-LM 兼容快照，不要强行训练视觉结构 |
| `no configured LoRA target suffix` | Transformers/模型结构变化导致层名不匹配 | 先 dry-run 查看真实层名；更新配置并记录新实验版本 |
| `assistant target was truncated` | `cutoff_len=512` 太短或样本过长 | 检查 chat template，必要时提高 cutoff 并重新评测显存/指标 |
| CUDA OOM | batch、上下文或 checkpoint 占用过高 | 保持 micro batch=1，开启 gradient checkpointing，确认没有把模型移到 CPU/多卡错误配置；必要时先降低上下文并重新验收 |
| `final holdout leakage detected` | 训练候选与 stress 精确或近重复 | 删除/改写泄漏样本，不降低阈值绕过保护 |
| 输出目录非空 | merge 为防覆盖拒绝运行 | 使用新的空版本目录，不删除未经备份的旧工件 |
| JSON invalid 或 reason 过长 | prompt、模型、token 截断或数据目标不一致 | 查看 raw predictions，先修契约/数据，再重训；不要只放宽解析 |
| Transformers P95 超时 | GPU、模型加载精度或并发过高 | 先单并发测量，固定 BF16，设置服务队列/超时/熔断 |
| Ollama 导入失败 | GGUF/架构/转换版本不兼容 | 保留 merged safetensors，优先使用隔离 Transformers 服务，不修改现有 Ollama tag |
| 主项目调用后回退规则 | URL、TLS、超时或 JSON 契约失败 | 检查 `/health`、`/assess`、证书、endpoint 白名单和服务日志；回退是预期安全行为 |

## 13. 历史版本和不可复现范围

- 第一版～第六版报告保留在 `reports/` 摘要中，磁盘旧 checkpoint 多数已清理；历史结果不能只凭目录名推断。
- 旧 v1/v3/v4/v5 数据和脚本用于 legacy 复现，不是当前 v9 推荐入口。
- 旧版模型评测必须绑定训练时的 prompt contract；不能用 v2 prompt 重新解释旧版数字。
- v9 是提示词 v2、`risk_sft_v9` 和当前冻结 stress 口径的组合结果。
- Ollama `qwen3.5:2b` 的精确上游转换 provenance 未验证；同族、同规模不等于已证明同一权重来源。
- stress 87 与规则/测试有共同设计背景，不等同于临床有效性证明；外部专家审查和独立数据仍是上线条件。

## 14. 可复现清单

开始训练前确认：

- [ ] AegisTraining commit 已记录；
- [ ] 主项目 fixture commit 已记录；
- [ ] `AEGIS_TRAINING_ROOT`、`AEGIS_PROJECT_ROOT`、`AEGIS_PROJECT_CORPUS` 已记录；
- [ ] Python、PyTorch、CUDA、Transformers、PEFT、bitsandbytes 版本已记录；
- [ ] 基座 revision、snapshot 文件和 gate 输出已保存；
- [ ] `risk_qlora_4060.yaml` 的 SHA-256 已保存；
- [ ] 数据 `manifest.json`、泄漏拒绝数、train/dev/test 行数已检查；
- [ ] dry-run 输出和 GPU 型号已保存；
- [ ] 训练、merge、eval 命令和输出目录已记录；
- [ ] 原始评测 JSON、raw predictions、训练 manifest 已备份；
- [ ] release checklist、哈希、回滚版本和审批状态已完成。

## 15. 入口速查

```text
training/configs/risk_qlora_4060.yaml          参数源
training/scripts/prepare_risk_sft_v4.py        当前 v9 数据构建
training/scripts/train_risk_qlora.py           dry-run / QLoRA 训练
training/scripts/merge_risk_qlora.py           adapter 合并
training/scripts/eval_risk_qlora.py            冻结 stress 评测
training/scripts/serve_risk_qlora.py           HTTP 推理服务（详见 SERVICE-RUNBOOK.md）
training/src/aegis_training/base_model_gate.py 官方基座 gate
training/src/aegis_training/data_contract.py   标签和 SFT 契约
training/src/aegis_training/leakage_guard.py    最终 holdout 泄漏保护
training/src/aegis_training/metrics.py         风险、JSON、延迟指标
reports/TRAINING-HISTORY-INDEX.md              版本谱系
reports/V9-TRAINING-EVAL-SUMMARY.md            当前 v9 验收摘要
aegis-psych-agent/docs/qlora-finetuning.md     主项目对应说明
```
