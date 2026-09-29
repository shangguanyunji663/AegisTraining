# Aegis 风险 QLoRA 训练操作手册

> **本手册与 [LEARNING-GUIDE.md](LEARNING-GUIDE.md) 的分工**：本手册只管**怎么操作**——环境搭建、完整命令、参数速查、验收门槛、发布回滚与故障排查。想**搞懂**——概念从零讲起、设计动机、易混淆点集中澄清、练习与自测——请读 [LEARNING-GUIDE.md](LEARNING-GUIDE.md)（新手从那里开始，遇到"为什么"随时跳过去查）。
>
> 训练仓库与生产应用必须保持隔离：训练代码和 GPU 依赖留在 AegisTraining，生产 FastAPI 进程只通过受保护的 HTTP/JSON 契约调用已验收模型服务。
>
> 相关文档：
> - **学习手册（概念与易混淆点）**：[training/LEARNING-GUIDE.md](LEARNING-GUIDE.md)
> - 数据格式与标签规则：[training/data/README.md](data/README.md)
> - 数据补充计划：[training/docs/DATA-SUPPLEMENT-PLAN.md](docs/DATA-SUPPLEMENT-PLAN.md)
> - 训练谱系：[reports/TRAINING-HISTORY-INDEX.md](../reports/TRAINING-HISTORY-INDEX.md)
> - 当前 v9 验收：[reports/V9-TRAINING-EVAL-SUMMARY.md](../reports/V9-TRAINING-EVAL-SUMMARY.md)
> - 模型发布模板：[docs/MODEL-RELEASES.md](../docs/MODEL-RELEASES.md)
> - v9 发布记录（当前 release-candidate）：[docs/V9-RELEASE-RECORD.md](../docs/V9-RELEASE-RECORD.md)
> - 生产项目接入说明：[aegis-psych-agent/docs/qlora-finetuning.md](https://github.com/shangguanyunji663/aegis-psych-agent/blob/main/docs/qlora-finetuning.md)

## 目录

- [第 1 章 工程边界和安全原则](#第-1-章-工程边界和安全原则)
- [第 2 章 当前推荐版本](#第-2-章-当前推荐版本)
- [第 3 章 目录和路径约定](#第-3-章-目录和路径约定)
- [第 4 章 隔离环境](#第-4-章-隔离环境)
- [第 5 章 基座模型 gate](#第-5-章-基座模型-gate)
- [第 6 章 数据契约和数据构建](#第-6-章-数据契约和数据构建)
- [第 7 章 QLoRA 参数](#第-7-章-qlora-参数)
- [第 8 章 训练流程](#第-8-章-训练流程)
- [第 9 章 v9 验收门槛和结果](#第-9-章-v9-验收门槛和结果)
- [第 10 章 主项目接入契约](#第-10-章-主项目接入契约)
- [第 11 章 发布、校验和回滚](#第-11-章-发布校验和回滚)
- [第 12 章 故障排查](#第-12-章-故障排查)
- [第 13 章 历史版本和不可复现范围](#第-13-章-历史版本和不可复现范围)
- [第 14 章 可复现清单](#第-14-章-可复现清单)
- [第 15 章 入口速查](#第-15-章-入口速查)
- [附录 操作易错点汇总清单](#附录-操作易错点汇总清单)

## 通用记号

- 所有命令均为 **Windows `cmd.exe` 语法**：`set` 设置环境变量，`^` 是行尾续行符。PowerShell 用户把 `^` 换成反引号 `` ` ``、`set X=Y` 换成 `$env:X="Y"`。
- `D:\AegisTraining` 是示例采用的训练根目录；按实际 checkout 位置替换。
- `> **易错点**：` 开头的引用块是历史踩坑记录，动手前务必阅读。
- `§X.Y` 表示"第 X.Y 节"；源码引用形如 `文件名.py:行号`，行号随重构可能漂移，以函数名为准。
- 全书术语与 [LEARNING-GUIDE.md](LEARNING-GUIDE.md) 术语口径一致：**基座模型**（被微调的原始模型）、**adapter**（LoRA 产出的小权重文件）、**合并/merge**（adapter 并入基座）、**冻结 holdout（stress 87）**（永不参与训练的验收集）、**评测**（跑出指标）与**验收**（对照门槛下结论）严格区分。

---

## 第 1 章 工程边界和安全原则

### 1.1 训练生命周期总览

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

### 1.2 六条红线

- **不把训练依赖追加到生产项目的 `requirements.txt`。**（生产镜像被 torch/CUDA 拖大数 GB，部署被训练变更耦合）
- **不把模型权重、HF cache、checkpoint、merged 权重、GGUF、日志或原始敏感数据提交到 Git。**（单基座约 4.5GB；敏感数据入 Git 等于不可撤销外泄）
- **不把 Ollama 的 Q8 GGUF 推理工件当作 Transformers/PEFT 的训练基座。**（无法证明与官方 revision 同源；详见第 5.3 节）
- **规则风险通道永久保留；QLoRA 只能升级风险，不能降低规则风险。**（融合 `max(规则, 模型)`，见第 10.2 节）
- **未通过冻结验收、外部审阅和发布清单的模型只能作为研究资产（research-only）。**（第 9 章）
- **心理健康和自杀风险数据必须完成授权、脱敏、许可证、用途和公开范围审查。**

> **易错点**：最常见的越界冲动是"临时调试方便"——把 `torch` 加进生产 requirements、把 checkpoint 拷进 Git LFS、把本机 8301 服务写进生产 `.env`。三者都违反红线。临时产物放在训练根目录的被忽略目录（`checkpoints/`、`exports/`、`hf-cache/`），本机服务只做 smoke test（第 8.6 节）。

设计动机的完整讲解见学习手册第二部分站 1 与站 10。

---

## 第 2 章 当前推荐版本

### 2.1 版本速览

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

### 2.2 要点提示

- **模型 / 数据集 / 提示词契约是一个整体**：换任何一个都必须重训并重新验收，旧数字不能拼给新组合（第 13 章）。
- **固定 revision**：保证下载的基座与验收时的逐字节一致（哈希核对见第 11 章）。
- **最优 checkpoint epoch 2 / step 360**：3 个 epoch 的 eval_loss 是 0.0818 → 0.0638 → 0.0654，第 3 个 epoch 回升（过拟合迹象），脚本自动回选 epoch 2。曲线解读见学习手册混 10。
- **`release-candidate` ≠ 已上线**：生产接入需显式审批并完成第 10、11 章动作。

### 2.3 版本编号遗留问题

版本编号有历史遗留：磁盘目录可能出现 `v3`、`v5`、`v7`、`v8` 等旧编号，正式谱系以 [TRAINING-HISTORY-INDEX.md](../reports/TRAINING-HISTORY-INDEX.md) 为准（旧 v1=第一版、旧 v3=第二版、旧 v4=第三版、旧 v5=第四版、旧 v7=第五版、旧 v8=第六版、旧 v9=第七版；"v2"编号作废）。

> **易错点**：磁盘目录名 ≠ 版次。引用历史结论一律以谱系索引为准（学习手册混 12 有完整对照表）。

---

## 第 3 章 目录和路径约定

### 3.1 目录树

```text
AegisTraining/
├── training/
│   ├── configs/risk_qlora_4060.yaml       # 当前 4060/8GB QLoRA 参数
│   ├── scripts/                           # prepare/train/merge/eval/serve 入口
│   ├── src/aegis_training/                 # 数据契约、gate、泄漏、指标、路径工具
│   ├── LEARNING-GUIDE.md                  # 学习手册（概念与易混淆点）
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

三类目录性质不同：`training/` 下的代码与配置进 Git（可审计）；`models/ data/ checkpoints/ exports/ envs/` 是本地工件（被 .gitignore 忽略）；`reports/ docs/` 只保留经审查的结论性文档。

### 3.2 路径守卫

脚本通过 `AEGIS_TRAINING_ROOT` 解析训练根目录，所有输入/输出必须位于训练根目录内。实现见 `training/src/aegis_training/paths.py` 的 `under(path, root, role)`：路径解析为绝对路径后 `relative_to(root)` 判断越界，越界抛 `path escapes allowed root`。这是防呆（`--output-root` 手滑写错立即报错）+ 安全（防止敏感输出被写到仓库外）双目的。

体验守卫（不依赖 GPU）：

```bat
D:\AegisTraining\envs\qlora-qwen35\Scripts\python.exe -c "import sys; sys.path.insert(0, r'D:\AegisTraining\training\src'); from pathlib import Path; from aegis_training.paths import under; print(under(Path(r'D:\AegisTraining\data\x.json'), Path(r'D:\AegisTraining'), 'demo'))"

D:\AegisTraining\envs\qlora-qwen35\Scripts\python.exe -c "import sys; sys.path.insert(0, r'D:\AegisTraining\training\src'); from pathlib import Path; from aegis_training.paths import under; under(Path(r'C:\Temp\x.json'), Path(r'D:\AegisTraining'), 'demo')"
```

第一条打印放行路径；第二条应看到 `ValueError: demo path escapes allowed root: ...`。

### 3.3 环境变量设置

```bat
set AEGIS_TRAINING_ROOT=D:\AegisTraining
set AEGIS_PROJECT_ROOT=D:\PythonProject\aegis-psych-agent
set AEGIS_PROJECT_CORPUS=%AEGIS_PROJECT_ROOT%\eval\fixtures\representative_corpus.json
set HF_HOME=%AEGIS_TRAINING_ROOT%\hf-cache
```

| 变量 | 作用 | 必需 |
| :--- | :--- | :--- |
| `AEGIS_TRAINING_ROOT` | 训练仓库与本地训练工件根目录（路径守卫基准） | 否（默认本 checkout） |
| `AEGIS_PROJECT_ROOT` | 生产仓库 checkout，只读其已提交的评测 fixture | 评测必需（第 8.4 节） |
| `AEGIS_PROJECT_CORPUS` | 版本化评测语料路径；也可用 `--corpus` 传入 | 数据构建必需 |
| `HF_HOME` | HF 下载缓存留在训练根内，不污染 C 盘 | 否 |

如果两个仓库不在同一台机器上，应把已审查、已版本化的 fixture 作为明确输入提供给训练流程，不要把本机绝对路径硬编码进数据文件或 manifest。

> **易错点**：`set` 只对当前 cmd 窗口有效，新开窗口要重设；忘设 `AEGIS_PROJECT_CORPUS` 时构建脚本会直接报 `--corpus is required`，这是设计行为，不要为绕过它指向工作区里未提交的语料副本。

### 3.4 模块地图（读代码前的依赖图）

```text
training/src/aegis_training/（公共库，被各脚本复用）
├── paths.py            路径守卫 training_root()/under()          （无内部依赖，所有脚本都用）
├── data_contract.py    RISK_SYSTEM_PROMPT 权威定义 + RiskSample/
│                       parse_sample()/normalize_message()/text_hash（无内部依赖）
├── base_model_gate.py  基座资格检查 verify_snapshot()/GateReport  （无内部依赖）
├── metrics.py          Prediction/classification_report()        （无内部依赖，纯标准库）
├── leakage_guard.py    最终 holdout 泄漏防护（legacy 管线）        ──依赖→ data_contract
└── source_ingest.py    外部/项目语料弱映射读取（legacy 管线）      ──依赖→ data_contract

training/scripts/
├── prepare_risk_sft_v4.py  v9 数据构建   ──依赖→ data_contract(仅 RISK_SYSTEM_PROMPT)、paths
│                                         （泄漏检查/标签裁决为脚本内联实现）
├── prepare_risk_sft.py     legacy v1 构建 ──依赖→ data_contract(全量校验)、leakage_guard、
│                                          source_ingest、paths
├── train_risk_qlora.py     训练          ──依赖→ paths（顶层）、base_model_gate（main 内导入）
├── merge_risk_qlora.py     合并          ──依赖→ base_model_gate、paths
├── eval_risk_qlora.py      冻结评测      ──依赖→ data_contract、metrics、paths
│                                         + 生产仓库 app.assessment（运行时导入）
├── eval_risk_devtest.py    devtest 评测  ──依赖→ paths + eval_risk_qlora 的推理器/解析器
└── serve_risk_qlora.py     推理服务      ──依赖→ data_contract、paths
```

四个设计事实：

1. **v9 管线刻意"轻依赖"**：`prepare_risk_sft_v4.py` 只取 prompt 与路径守卫，泄漏检查与标签裁决内联实现——它消费的 consolidated 候选格式与 `data_contract` 的原始契约格式是两套数据形态。**两套泄漏实现口径不同（纯 Jaccard vs max(Jaccard, SequenceMatcher)），不许混用**（学习手册混 13）。
2. **`leakage_guard.py` 与 `source_ingest.py` 属 legacy 管线**：只被 `prepare_risk_sft.py` 使用；保留作"数据安全"与"弱监督映射"的教学材料（走读见学习手册）。
3. **评测跨仓库拿规则基线**：`eval_risk_qlora.py` 运行时把 `AEGIS_PROJECT_ROOT` 指向的生产项目插入 `sys.path` 并 `from app.assessment import assess_message`——验收对比的就是生产同款规则引擎，因此评测必须设置该变量。
4. **devtest 评测复用冻结评测的推理器**：跨脚本导入保证同口径（同一 generate 参数、同一 JSON 解析逻辑）。

---

## 第 4 章 隔离环境

### 4.1 独立虚拟环境

训练依赖版本激进、更新频繁，与生产项目共存一个环境会互相升降级破坏。独立环境（`envs/qlora-qwen35/`）坏了重建即可。推荐 Python 3.11、与显卡驱动匹配的 CUDA PyTorch：

```bat
D:\AegisTraining\envs\qlora-qwen35\Scripts\python.exe -m pip install --upgrade pip
D:\AegisTraining\envs\qlora-qwen35\Scripts\python.exe -m pip install -r D:\AegisTraining\training\requirements-qlora.txt
```

### 4.2 依赖包速查

| 包 | 版本要求 | 作用 |
| --- | --- | --- |
| `transformers` | `>=5.2.0` | Qwen3.5 架构实现、`apply_chat_template`、Trainer |
| `peft` | `>=0.18.0` | LoRA：注入低秩模块、保存/加载 adapter、`merge_and_unload` |
| `bitsandbytes` | `>=0.49.0` | 4-bit NF4 量化线性层与 8-bit/分页优化器（QLoRA 的 "Q"） |
| `accelerate` | `>=1.12.0` | `device_map`、模型放置与加速 |
| `datasets` | `>=4.4.0` | JSONL 包装成可训练 Dataset |
| `safetensors` | `>=0.6.2` | 安全读写权重文件 |
| `PyYAML` | `>=6.0.2` | 读取唯一参数源 `risk_qlora_4060.yaml` |
| `scikit-learn` | `>=1.6.0` | 遗留/离线分析辅助；当前主链路未直接导入 |

> **易错点**：`>=` 是下限不是锁定。真正复现实验必须保留 `pip freeze` 输出（[REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) 第 3 节）。

### 4.3 PyTorch 与 CUDA

`pip install torch` 默认可能装 **CPU 版**——能 import 但用不了 GPU。需要：NVIDIA 驱动 → CUDA 运行时 → GPU 版 PyTorch wheel 三者匹配，按本机驱动从官方渠道单独安装。不能因为 `pip install -r` 成功就认为 CUDA 训练可用。

BF16/FP16 规则：`bnb_4bit_compute_dtype: bfloat16` 首选；仅当 GPU 不支持 BF16 且 `fp16_fallback: true` 时显式降级 FP16（降级是显式配置允许的，不是静默发生的）。两者的区别见学习手册混 4。

### 4.4 环境自检（训练前必做）

```bat
D:\AegisTraining\envs\qlora-qwen35\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

预期输出（示例）：

```text
2.x.x+cu12x
True
NVIDIA GeForce RTX 4060 Laptop GPU
```

版本号**必须带 `+cu12x` 后缀**（`+cpu` 即装错 wheel）；`True` 必须；第三行是显卡名。任何一行不符，先解决环境再继续。

> **易错点**：用错解释器是最常见事故——务必始终用 `envs\qlora-qwen35\Scripts\python.exe` 完整路径运行脚本。`cuda.is_available()` 为 False 的排查顺序：CPU 版 wheel → 驱动过旧（`nvidia-smi`）→ 非 NVIDIA 显卡。

---

## 第 5 章 基座模型 gate

训练只能使用固定 revision 的官方 Qwen3.5-2B-Base safetensors 快照（`Qwen/Qwen3.5-2B-Base` @ `b1485b2fa6dfa1287294f269f5fb618e03d52d7c`，`qwen3_5`，hidden 2048，24 层）。快照至少包含 `config.json` 和 `model.safetensors.index.json` 且权重清单非空。

执行 gate：

```bat
D:\AegisTraining\envs\qlora-qwen35\Scripts\python.exe ^
  D:\AegisTraining\training\src\aegis_training\base_model_gate.py ^
  --snapshot-dir "D:\AegisTraining\models\Qwen3.5-2B-Base"
```

### 5.1 检查项

- repo、revision、model type、架构、hidden size、层数、权重清单与钉死的官方值逐一比对；
- **必须是可训练 safetensors 而非 GGUF**（缺 index.json 通常意味着推理工件或不完整下载）；
- 拒绝视觉/多模态模块（模块名含 vision/visual/image/video）；
- Ollama 转换 provenance 可选：未提供证明时状态只能为 `pass_same_family_with_caveat`。

### 5.2 通过状态的两级语义（`base_model_gate.py:111`）

| 状态 | 含义 |
| --- | --- |
| `pass_same_family_with_caveat` | **常规通过**：结构、架构、规模全部匹配官方快照；Ollama provenance 未验证。当前 v9 管线（不传 provenance）**恒为此状态** |
| `pass_exact` | 仅在提供经验证的 Ollama 转换证明时出现；当前本机没有该证明，不应强行通过严格模式 |

dry-run 输出必须能看到 `pass_same_family_with_caveat`——**不要把它当成 gate 失败**（详解见学习手册混 11）。

### 5.3 GGUF 与 safetensors 的区别

| 维度 | safetensors（训练用） | GGUF（推理工件） |
| --- | --- | --- |
| 生态 | Hugging Face / Transformers / PEFT | Ollama / llama.cpp |
| 内容 | 与官方发布逐字节对应的原始权重 | 可能经过再量化、重排与打包 |
| 溯源 | revision + SHA-256 精确锁定 | 通常只能追溯到"某个转换" |
| 能否作 QLoRA 基座 | ✅ 唯一接受格式 | ❌ 禁止（第 1 章红线） |

> **易错点**：从非官方镜像下载"同名字"模型被 gate 拒绝，是保护而不是麻烦。

---

## 第 6 章 数据契约和数据构建

> 本章命令与核对项为主；契约字段逐个解释、标签原则与边界案例见学习手册站 2~4，裁决漏斗与构建脚本源码走读见学习手册文末附录「源码走读精选」。

### 6.1 原始数据契约（校验摘要）

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

校验规则（`data_contract.py` 强制执行，违反即抛 `DataContractError` 终止）：

- `risk_level` 只能是 `low` / `medium` / `high`；
- `reason` ≤ 20 字符；原始 `message` ≤ 1500 字符；
- `speaker_scope` 只能是 `self` / `third_party` / `fictional`；
- **`high` 必须 `speaker_scope=self`**（描述说话人自身风险）；
- `sample_id` 不得重复；
- `label_method` / `review_status` 必须如实记录，自动映射不得伪装成人工审核。

体验校验（不依赖 GPU）：

```bat
D:\AegisTraining\envs\qlora-qwen35\Scripts\python.exe -c "import sys; sys.path.insert(0, r'D:\AegisTraining\training\src'); from aegis_training.data_contract import parse_sample; s = parse_sample({'sample_id':'demo-001','message':'我最近觉得自己很多余。','risk_level':'medium','reason':'明显痛苦但无自伤意向','source':'demo','speaker_scope':'self','label_method':'manual_reviewed','review_status':'manual_reviewed'}); print('OK:', s.sample_id, s.risk_level)"
```

标签原则与边界案例对照（high/medium/low 的分界、"第三人称不升自身风险"、`"不配被爱"medium ↔ "不配活着"high` 的细线）见学习手册混 7、混 8 与站 2。

### 6.2 当前数据来源和规模

当前主流程使用已审查的 `training/data/consolidated_risk_v1`，经 `prepare_risk_sft_v4.py` 生成 `data/risk_sft_v9`：

| 输出 | 当前规模 | 用途 |
| --- | ---: | --- |
| `train.jsonl` | 2867 | 训练 |
| `dev.jsonl` | 200 | epoch 评估、early stopping 和最优模型选择 |
| `test.jsonl` | 1414 | devtest 参考评测，**不作为最终冻结验收集** |
| 项目 `base` | 63 | 可作为开发候选 |
| 项目 `stress` | 87 | 永久冻结最终 holdout，不能进 train/dev |

### 6.3 提示词契约 v2

当前 system prompt 必须和训练、评测、推理服务保持一致（全文见 `training/src/aegis_training/data_contract.py` 的 `RISK_SYSTEM_PROMPT`，该文件是权威定义）：

- **两处独立物理副本必须逐字同步**：训练仓库 `data_contract.py`（权威）↔ 生产项目 `app/llm/client.py`。数据构建/冻结评测/本地服务均从 `data_contract.py` 导入，自动一致。
- `build_consolidated.py` 内嵌的是构建期旧 prompt，仅作审计产物，**其输出不得直接当训练数据**（学习手册混 14）。
- **改 prompt 是重训级事件**：新 prompt → 重建数据 → 重训 → 全量重验收；旧版数字只绑定旧 prompt（学习手册混 15）。

### 6.4 泄漏防护（操作要点）

`prepare_risk_sft_v4.py` 内联实现（与 legacy `leakage_guard.py` 口径不同，不许混用）：

1. 读取 stress 87 条并**断言数量准确**（不是 87 直接报错）；
2. 候选文本 NFKC + 去空白 + 转小写规范化（防换字符绕过）；
3. 精确哈希拒绝原文复用；
4. 字符 3-gram Jaccard ≥ `0.82` 拒绝近重复（阈值是"漏检"与"误杀"的权衡值，**不允许调低绕过**）；
5. 拒绝记录写入 manifest `leakage_rejected`，逐条可审计；
6. train 内部与 dev 也去重。

明确禁区：`risk.json`、`routing.json`、`multi_turn_corpus.json`、`safety.json`、RAG fixtures、Harness、probe、政策文档、Skill 内容和测试消息**不得作为训练语料**。

同时如实标注边界：stress 是工程冻结集，不是完全独立的临床盲测集；上线前仍需外部专家审核和独立测试。

### 6.5 可选：重建 consolidated 候选池

只有需要重建候选池或审计来源时才运行：

```bat
set AEGIS_CONSOLIDATED_ROOT=D:\AegisTraining\training\data\consolidated_risk_v1
python D:\AegisTraining\training\data\consolidated_risk_v1\build_consolidated.py
```

重建前必须记录：项目 `representative_corpus.json`（冻结输入保持当前版本）；PsySUICIDE、suicide 原始数据和 `metaphor_corpus_v1.jsonl` 的来源、许可证、获取时间和 SHA-256；三个环境变量；外部数据审查结果。其输出是**旧 prompt 审计中间产物**，当前 v9 仍必须经 `prepare_risk_sft_v4.py` 重新裁决生成。

### 6.6 构建当前数据与 manifest 检查

#### 默认构建参数

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--train-quota` | `1000 1000 1000` | high/medium/low 目标配额（含人工样本的上限） |
| `--dev-quota` | `70 60 70` | dev 配额，合计 200 |
| `--max-text-chars` | `300` | 文本长度上限（配合 cutoff_len=512 保证 assistant 不被截断） |
| `--leak-threshold` | `0.82` | 与 stress 的近重复阈值 |
| `--medium-upsample-cap` | `1.35` | medium 扩量上限 |
| `--distill-limit` | `200` | 蒸馏来源挖掘上限 |
| 随机种子 | `42` | 确定性哈希抽样，同输入必得同输出 |

#### 构建命令

```bat
D:\AegisTraining\envs\qlora-qwen35\Scripts\python.exe ^
  D:\AegisTraining\training\scripts\prepare_risk_sft_v4.py ^
  --consolidated-root "D:\AegisTraining\training\data\consolidated_risk_v1" ^
  --corpus "%AEGIS_PROJECT_CORPUS%" ^
  --output-root "D:\AegisTraining\data\risk_sft_v9"
```

#### 构建后必须检查 manifest

```bat
python -c "import json; p=r'D:\AegisTraining\data\risk_sft_v9\manifest.json'; m=json.load(open(p,encoding='utf-8')); print(m['schema_version']); print(m['train']); print(m['dev']); print(m['devtest']); print('leaks=',len(m['leakage_rejected']))"
```

预期 `schema_version` 为 `risk_sft_v9`；train/dev/devtest 数量与 6.2 节表格对上；`leakage_rejected` 逐条审阅（每条带 ID 与拒绝原因 `exact` 或 `ngram:序号`）——审阅它是确认"防护在工作"。manifest 同时记录样本哈希、来源、映射方式、分布、`relabel_summary`、hard negatives 与 medium 扩量数量，是第 14 章清单与第 11 章发布元数据的直接输入。

> **易错点**：
> 1. 构建退出码 0 就跳过 manifest 审阅——退出码只说明程序没崩，不说明数据质量合格。
> 2. 手工编辑 `train.jsonl` 加私货——绕过契约、泄漏检查和 manifest，审计链作废。要加数据走 `authored/` 与构建参数。

---

## 第 7 章 QLoRA 参数

唯一参数源是 `training/configs/risk_qlora_4060.yaml`。变更配置后应保存新的 manifest、checkpoint 和验收报告；做实验复制为版本化文件（如 `risk_qlora_exp01.yaml`）显式传入 `--config`，**不要直接改当前配置**。

### 7.1 基座

```yaml
base_model:
  repo_id: Qwen/Qwen3.5-2B-Base
  revision: b1485b2fa6dfa1287294f269f5fb618e03d52d7c
  trust_remote_code: false   # 不执行模型仓库自定义加载代码；官方 Qwen3.5 已被原生支持
  text_only: true
```

### 7.2 4-bit 量化参数

| 参数 | 值 | 目的 |
| --- | --- | --- |
| `load_in_4bit` | `true` | 低显存加载基座 |
| `bnb_4bit_quant_type` | `nf4` | NormalFloat4 量化 |
| `bnb_4bit_use_double_quant` | `true` | 二重量化 |
| `bnb_4bit_compute_dtype` | `bfloat16` | 计算精度（"4-bit 存、BF16 算"，见学习手册混 3）；不支持时 fallback fp16 |

四项合起来是 QLoRA 论文的完整配方；改动任何一项都改变数值行为，需重新实验与验收。

### 7.3 LoRA 参数

| 参数 | 值 |
| --- | --- |
| `rank` / `r` | `8` |
| `alpha` / `lora_alpha` | `16` |
| `dropout` | `0.05` |
| `bias` | `none` |
| `task_type` | `CAUSAL_LM` |
| 目标模块 | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`, `in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, `out_proj` |

目标模块与实际量化文本模型的线性层后缀**取交集**：配置里写了但模型里没有的层被跳过；一个都匹配不上（说明拿错模型或结构变了）报 `no configured LoRA target suffix` 失败。视觉/图像/视频模块被排除。

### 7.4 Trainer 参数

| 参数 | 值 | 参数 | 值 |
| --- | --- | --- | --- |
| `cutoff_len` | `512` | `optim` | `paged_adamw_8bit` |
| `per_device_train_batch_size` | `1` | `max_grad_norm` | `1.0` |
| `per_device_eval_batch_size` | `1` | `logging_steps` | `10` |
| `gradient_accumulation_steps` | `16` | `eval_strategy` | `epoch` |
| `gradient_checkpointing` | `true` | `save_strategy` | `epoch` |
| `learning_rate` | `0.0001` | `save_total_limit` | `2` |
| `weight_decay` | `0.0` | `seed` | `42` |
| `num_train_epochs` | `3` | `bf16` | `true` |
| `warmup_ratio` | `0.05` | `fp16_fallback` | `true` |
| `lr_scheduler_type` | `cosine` | `early_stopping_patience` | `2` |
| 最优指标 | `eval_loss` | `load_best_model_at_end` | `true` |
| `report_to` | `none` | | |

参数的逐条设计理由（显存组/优化组/节奏组）与"为什么用 eval_loss 选型"见学习手册站 6。

### 7.5 loss 掩码与关键实现细节

- `apply_chat_template` 对 system/user 的 labels 写 `-100`，**只对 assistant 回复计算 loss**；assistant 目标被 `cutoff_len` 截断时**直接报错**（`assistant target was truncated`），绝不静默训练空目标。
- `model.config.use_cache = False`：KV 缓存与梯度检查点不兼容。
- `prepare_model_for_kbit_training(...)`：量化模型进训练前的标准准备，跳过会导致 4-bit 层训练异常。
- `DataCollatorForSeq2Seq(padding=True, label_pad_token_id=-100)`：动态 pad，pad 位置不参与 loss。
- `device_map={"": 0}`：整模型放 0 号 GPU；`tokenizer.pad_token = tokenizer.eos_token`；`remove_unused_columns=False`。

### 7.6 步数推算

```text
有效批量  = 1 × 16 = 16
每 epoch  = ceil(2867 / 16) = 180 步
总步数   = 180 × 3 = 540 步（与 v9 实测一致）
warmup   ≈ 5% × 540 ≈ 27 步
```

> **易错点**：任何超参变更（rank、batch、epoch……）都是**新实验**：版本化配置文件 + 新 `--output-root` + dry-run + 完整重新验收。

---

## 第 8 章 训练流程

### 8.1 Dry-run（先跑，3 分钟排掉 3 小时的雷）

```bat
D:\AegisTraining\envs\qlora-qwen35\Scripts\python.exe ^
  D:\AegisTraining\training\scripts\train_risk_qlora.py ^
  --data-root "D:\AegisTraining\data\risk_sft_v9" ^
  --snapshot-dir "D:\AegisTraining\models\Qwen3.5-2B-Base" ^
  --dry-run
```

输出至少确认六项并抄进实验记录：CUDA 设备名；实际 `compute_dtype`；实际匹配的 target modules；`train_rows`/`dev_rows`（应与 manifest 一致：2867/200）；`trainable_params`/`total_params`/`trainable_percent`（8,409,600 / 1,890,234,688 / ≈0.4449%）；gate 状态 `pass_same_family_with_caveat`。

### 8.2 正式训练

```bat
D:\AegisTraining\envs\qlora-qwen35\Scripts\python.exe ^
  D:\AegisTraining\training\scripts\train_risk_qlora.py ^
  --data-root "D:\AegisTraining\data\risk_sft_v9" ^
  --snapshot-dir "D:\AegisTraining\models\Qwen3.5-2B-Base" ^
  --output-root "D:\AegisTraining\checkpoints\aegis-risk-qwen3.5-2b-v9"
```

**必须显式传入版本化 `--output-root`**：配置默认目录不带版本号；两个实验共用一个输出目录 = adapter 互相覆盖、谱系灾难。需要改参数时复制配置为版本化实验文件并 `--config` 显式传入。

训练产物结构：

```text
checkpoints/aegis-risk-qwen3.5-2b-v9/
├── adapter/                        # 最终 adapter（自动回选最优 checkpoint 后保存）
├── checkpoint-360/                 # epoch 2 训练状态（可恢复训练）
├── checkpoint-540/                 # epoch 3 训练状态
└── training-manifest.json          # gate 结果、dtype、目标层、参数量、行数、峰值显存
```

v9 实际记录：eval_loss 0.0818 → 0.0638 → 0.0654，最优 epoch 2 / step 360；540 steps，约 2h54m，峰值 CUDA 显存 5,072,041,472 bytes（约 4.9GB）。曲线解读见学习手册混 10。

### 8.3 合并 adapter

合并前 adapter 目录须含 `adapter_config.json` 与 `adapter_model.safetensors`；**输出目录必须为空或不存在**（防覆盖保护）：

```bat
D:\AegisTraining\envs\qlora-qwen35\Scripts\python.exe ^
  D:\AegisTraining\training\scripts\merge_risk_qlora.py ^
  --snapshot-dir "D:\AegisTraining\models\Qwen3.5-2B-Base" ^
  --adapter-dir "D:\AegisTraining\checkpoints\aegis-risk-qwen3.5-2b-v9\adapter" ^
  --output-dir "D:\AegisTraining\exports\aegis-risk-qwen3.5-2b-v9-merged"
```

合并行为：BF16 加载官方基座（与验收/服务口径一致）→ `PeftModel.from_pretrained` 加载 adapter → `merge_and_unload(safe_merge=True)` 数值并入（含异常值检查）→ 分片 safetensors 保存（默认 2GB/片）→ 确定性生成 `do_sample=false` → 写入 `aegis-export-manifest.json`。

**为什么不直接在生产加载 adapter**：要求生产进程装 peft/bitsandbytes 并承担 4-bit 路径，违反隔离红线，且验收口径（BF16 merged）与生产口径分叉。

### 8.4 冻结集评测

> **前置条件**：评测要跑规则基线，运行时从 `AEGIS_PROJECT_ROOT` 导入生产同款 `assess_message`。必须先设置，否则报 `AEGIS_PROJECT_ROOT is required for rules baseline evaluation`；这是刻意设计，没有跳过规则基线的开关。

```bat
set AEGIS_PROJECT_ROOT=D:\PythonProject\aegis-psych-agent

D:\AegisTraining\envs\qlora-qwen35\Scripts\python.exe ^
  D:\AegisTraining\training\scripts\eval_risk_qlora.py ^
  --original-model qwen3.5:2b ^
  --qlora-model-dir "D:\AegisTraining\exports\aegis-risk-qwen3.5-2b-v9-merged" ^
  --timeout 8 ^
  --max-new-tokens 64 ^
  --output "D:\AegisTraining\reports\risk-qlora-eval-v9.json"
```

脚本流程：只取 fixture 的 `layer=stress` 87 条 → 同时测规则基线、原始模型、QLoRA 原始输出 → 计算 `rules ∪ original` 与 `rules ∪ qlora` → JSON 有效率、合法标签率、reason 长度、accuracy、macro-F1、每类 precision/recall/F1、high recall、non-high→high FPR、medium→high rate、P95 → 按 base/stress、implicit high、direct high、third person 分层 → 保存 raw predictions。

**为什么要测五个组合**：`rules_union_qlora` 才是生产等效指标（融合口径）；`qlora_raw` 用于诊断模型本身；`rules_union_original` 回答"微调带来多少增益"。八项门槛全部定义在融合口径上（为什么，见学习手册站 9）。

指标公式速查（实现见 `metrics.py`）：

- `precision = TP/(TP+FP)`；`recall = TP/(TP+FN)`；`F1 = 2PR/(P+R)`；
- `macro-F1` = 三类 F1 简单平均（不按样本数加权）；
- `non_high→high FPR` = 误升 high 的 non-high 数 ÷ 全部 non-high；
- `P95` = 87 条延迟排序取第 95 百分位。

`--qlora-model`（Ollama tag）与 `--qlora-model-dir`（Transformers 目录）**互斥**，口径不同不能混比；当前推荐 Transformers 路径。

体验指标计算（纯标准库）：

```bat
D:\AegisTraining\envs\qlora-qwen35\Scripts\python.exe -c "import sys; sys.path.insert(0, r'D:\AegisTraining\training\src'); from aegis_training.metrics import Prediction, classification_report; rows=[Prediction('a','high','high'),Prediction('b','medium','high'),Prediction('c','low','low'),Prediction('d','high','high'),Prediction('e','medium','medium')]; r=classification_report(rows); print('accuracy=',r['accuracy']); print('high_recall=',r['high_recall']); print('non_high_to_high_fpr=',r['non_high_to_high_fpr'])"
```

预期：accuracy 0.8；high recall 1.0；FPR = 1/3 ≈ 0.333。

### 8.5 devtest 参考评测

`--model-dir/--test-jsonl/--output` 必须显式传 v9 路径（脚本 argparse 默认值指向旧 v4 目录）：

```bat
D:\AegisTraining\envs\qlora-qwen35\Scripts\python.exe ^
  D:\AegisTraining\training\scripts\eval_risk_devtest.py ^
  --model-dir "D:\AegisTraining\exports\aegis-risk-qwen3.5-2b-v9-merged" ^
  --test-jsonl "D:\AegisTraining\data\risk_sft_v9\test.jsonl" ^
  --output "D:\AegisTraining\reports\risk-qlora-devtest-v9.json" ^
  --max-new-tokens 64
```

输出 accuracy、macro-F1、每类 F1、FPR、medium→high rate、reason/JSON 有效率和 P95，保留最多 40 条误判样本。两份诊断材料：**混淆矩阵**（gold×pred）与 **errors_sample**。口径细节：解析失败按协议**回退记 low** 参与 confusion——`json_valid_rate` 必须与 accuracy 一起读。devtest 用于开发诊断，**不能替代冻结 stress 87 的上线 gate**。

### 8.6 推理服务（smoke test）

```bat
set AEGIS_QLORA_MODEL_DIR=D:\AegisTraining\exports\aegis-risk-qwen3.5-2b-v9-merged
D:\AegisTraining\envs\qlora-qwen35\Scripts\python.exe ^
  D:\AegisTraining\training\scripts\serve_risk_qlora.py ^
  --model-dir "%AEGIS_QLORA_MODEL_DIR%" ^
  --host 127.0.0.1 --port 8301
```

可选 `--load-4bit` 降低推理显存，但可能改变边界样本预测；验收采用 BF16，**切换 4-bit 前必须重新评测**。

接口：`GET /health`、`POST /assess`。

```bat
curl.exe http://127.0.0.1:8301/health
curl.exe -X POST http://127.0.0.1:8301/assess ^
  -H "Content-Type: application/json" ^
  -d "{\"message\":\"我最近考试压力很大，晚上睡不着\"}"
```

预期响应：`{"status":"ok","calls":0}`；`{"risk_level":"medium","reason":"睡眠困扰和学业压力","latency_ms":912.0}`。

服务端行为：与验收相同的 v2 prompt、`temperature=0`、`max_new_tokens=64`、宽容 JSON 解析（剥围栏 → 标准 loads → 正则提取 `{...}` → 校验标签 → reason 截 120 字；**任何一步失败返回 null，绝不猜标签**）。

**错误语义**：请求非法/空 message → HTTP 400；推理异常 → HTTP 500（进程不退出）；未知路径 → 404；输出无法解析 → **HTTP 200 + `risk_level: null`**。null 不是"低风险"，调用方必须回退规则（学习手册混 6）。

> ⚠️ 本服务是**受控隔离环境的模型端点**，不是生产服务：无 TLS、认证、限流与审计。生产部署要求见 [SERVICE-RUNBOOK.md](docs/SERVICE-RUNBOOK.md)。

---

## 第 9 章 v9 验收门槛和结果

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

八项必须**全部**通过（缺一即整体失败；每个门槛防什么事故见学习手册站 9）。

辅助指标：stress overall accuracy `0.782`、medium recall `0.882`、third-person accuracy `0.818`；devtest 1414 参考：accuracy `0.895`、high-F1 `0.886`、FPR `0.050`。

> **如实说明边界**：devtest 上 medium-F1 0.380 低于上一版（0.496）。冻结集并非独立盲测集（合成表达与仓库规则/测试有共同设计背景）；生产结论仍需外部专家审阅与完全隔离的测试集。

### 9.1 门槛失败怎么办

1. 不注册生产服务或 Ollama tag；
2. 记录完整 manifest、配置、commit、环境和原始评测报告；
3. 将 adapter 标记为 research-only；
4. 保留上一版已批准模型或规则通道；
5. 先分析分层误判和提示词/数据变化，再决定下一轮训练。

> **易错点**：门槛是**人工**对照报告确认的，评测脚本不自动输出 pass/fail（REPRODUCIBILITY 第 7 节）。看到指标好就宣布"通过验收"而不逐项对照记录，是验收环节最常见的偷工。

---

## 第 10 章 主项目接入契约

### 10.1 生产配置与开关语义

```ini
RISK_QLORA_ENABLED=false
RISK_QLORA_URL=https://qlora-endpoint.example.invalid
RISK_QLORA_TIMEOUT_SECONDS=8
```

- `RISK_QLORA_ENABLED` 默认 `false`，未明确启用时行为不变；
- 主项目通过 `POST /assess` 调用，不在 FastAPI 进程中加载训练模型；
- 规则评估永久执行，融合 `max(规则风险, QLoRA 风险)`——QLoRA 不能把规则的 high 降为 medium/low；
- 超时、网络错误、HTTP 错误、JSON 非法或 `risk_level` 缺失都回退规则；
- 主项目 URL 校验拒绝 localhost、环回、私有和保留地址；生产使用经审批的公网 HTTPS endpoint；
- 高风险报告、审批、工具调用和安全模板不由模型直接决定，仍由主项目业务链路控制。

### 10.2 max 融合要点

```text
low < medium < high（数值化 1/2/3）
融合结果 = max(模型结果, 规则结果)   # 逐条取较高者
```

两个通道各自只做"加分项"：模型为规则补盲区，规则为模型兜底线；任何单通道失效，结果至少不差于纯规则。想让模型"纠偏"规则判严的个案在设计上不可能——这类需求走规则策略修订。完整原理见学习手册站 10。

> **易错点**：生产 `.env` 里写 `http://127.0.0.1:8301` 做联调后忘改回——URL 校验会直接拒绝环回/私有地址（保护），联调用测试配置，不要找校验漏洞绕过。

---

## 第 11 章 发布、校验和回滚

### 11.1 发布元数据清单

每个 release candidate 必须具备：

- 模型名、版本和训练仓库 commit；
- 基座 repo、固定 revision、许可证和下载来源；
- 数据 manifest、数据来源授权、脱敏状态、`label_method` 和 `review_status`；
- system prompt/risk contract 版本；
- adapter 与 merged 文件的 SHA-256；
- 训练配置文件和 `training-manifest.json`；
- 冻结 holdout、外部测试集、延迟和显存结果；
- 生产接入状态、已知限制和回滚版本。

没有完整元数据的权重不视为可发布模型。Windows 计算哈希：

```powershell
Get-FileHash D:\AegisTraining\exports\aegis-risk-qwen3.5-2b-v9-merged\*.safetensors -Algorithm SHA256
```

发布元数据填写 [docs/MODEL-RELEASES.md](../docs/MODEL-RELEASES.md)。权重放受控对象存储、Hugging Face 或经审查的 Release，不放回代码仓库。

### 11.2 回滚

回滚顺序：**关闭 `RISK_QLORA_ENABLED` → 恢复上一版已批准 endpoint/model → 验证规则和 API → 保存回滚事件记录**。第一步永远是拉回已知安全状态，诊断放在模型下线之后。回滚事件记回 [TRAINING-HISTORY-INDEX.md](../reports/TRAINING-HISTORY-INDEX.md)。

---

## 第 12 章 故障排查

### 12.1 方法论

全套流程分五个环节：**环境 → 数据 → 训练 → 评测 → 服务**。定位口诀：**报错发生在哪一步，就先确认上一步的产物是否健康**（数据没审 manifest 就训练、没跑 dry-run 就训 3 小时，都会把排障难度乘以十）。不要用"重跑一次"当第一反应——根因是泄漏或契约漂移时，重跑只是复现问题还烧 3 小时。

### 12.2 现象速查表

| 现象 | 常见原因 | 处理 |
| --- | --- | --- |
| `CUDA PyTorch is required` | CPU 版 PyTorch 或 CUDA 不可用 | 安装匹配驱动的 GPU wheel，先验证 `torch.cuda.is_available()` |
| base gate 缺 `model.safetensors.index.json` | GGUF、非官方快照或不完整下载 | 重新取得固定 revision 的官方 safetensors |
| gate 拒绝 vision modules | 加载了多模态路径 | 使用纯文本 causal-LM 快照 |
| `no configured LoRA target suffix` | 层名不匹配 | 先 dry-run 查看真实层名；更新配置并记录新实验版本 |
| `assistant target was truncated` | cutoff_len 太短或样本过长 | 检查 chat template，必要时提高 cutoff 并重新评测 |
| CUDA OOM | batch/上下文/配置错误 | 保持 micro batch=1 + gradient checkpointing，确认 device_map；必要时降上下文并重新验收 |
| `final holdout leakage detected` | 候选与 stress 精确或近重复 | 删除/改写泄漏样本，**不降低阈值绕过** |
| 输出目录非空 | merge 防覆盖拒绝 | 使用新的空版本目录 |
| JSON invalid 或 reason 过长 | prompt/模型/截断/数据目标不一致 | 查看 raw predictions，先修契约/数据再重训；不要放宽解析 |
| Transformers P95 超时 | 精度或并发问题 | 单并发测量、固定 BF16、服务队列/超时/熔断 |
| Ollama 导入失败 | GGUF/架构/转换版本不兼容 | 保留 merged safetensors，优先隔离 Transformers 服务 |
| 主项目调用后回退规则 | URL/TLS/超时/JSON 契约失败 | 检查 `/health`、`/assess`、证书、白名单与服务日志；回退是预期安全行为 |

### 12.3 高频问题深挖

- **训练正常但评测 JSON 有效率下跌**：先 diff 训练与评测的 prompt 版本（两处独立副本，第 6.3 节）；再看 raw predictions（围栏包裹？reason 超长截断？）。修复从契约/数据侧下手，禁止放宽解析"让数字好看"。
- **评测延迟远超验收记录**：确认没开 `--load-4bit`、无并发压测；单并发 BF16 是基线口径。长尾集中在个别长样本时查 message 长度分布。
- **主项目"看起来没接上模型"**：大概率是回退机制在正确工作。先看主项目日志回退原因（超时？400？null？），再到模型服务侧对时间戳。

---

## 第 13 章 历史版本和不可复现范围

- 第一版～第六版报告保留在 `reports/` 摘要中，磁盘旧 checkpoint 多数已清理；历史结果不能只凭目录名推断（版次映射见第 2.3 节）。
- 旧 v1/v3/v4/v5 数据和脚本用于 legacy 复现，不是当前 v9 推荐入口。
- 旧版模型评测必须绑定训练时的 prompt contract；不能用 v2 prompt 重新解释旧版数字。
- v9 是提示词 v2、`risk_sft_v9` 和当前冻结 stress 口径的组合结果。
- Ollama `qwen3.5:2b` 的精确上游转换 provenance 未验证；同族、同规模不等于已证明同一权重来源。
- stress 87 与规则/测试有共同设计背景，不等同于临床有效性证明；外部专家审查和独立数据仍是上线条件。

**潜台词**：结论只在产生它的条件下成立。"数据 + prompt + 基座 + 口径"四位一体，动任何一个都要重新走完全流程。

---

## 第 14 章 可复现清单

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

完整证据规范见 [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)。

---

## 第 15 章 入口速查

```text
training/LEARNING-GUIDE.md                     学习手册（概念与易混淆点）
training/configs/risk_qlora_4060.yaml          参数源
training/scripts/prepare_risk_sft_v4.py        当前 v9 数据构建
training/scripts/train_risk_qlora.py           dry-run / QLoRA 训练
training/scripts/merge_risk_qlora.py           adapter 合并
training/scripts/eval_risk_qlora.py            冻结 stress 评测
training/scripts/eval_risk_devtest.py          devtest 参考评测
training/scripts/serve_risk_qlora.py           HTTP 推理服务（详见 SERVICE-RUNBOOK.md）
training/src/aegis_training/base_model_gate.py 官方基座 gate
training/src/aegis_training/data_contract.py   标签和 SFT 契约（RISK_SYSTEM_PROMPT 权威定义）
training/src/aegis_training/leakage_guard.py    泄漏保护（legacy 管线使用；v9 为 prepare 脚本内联实现）
training/src/aegis_training/source_ingest.py    外部语料弱映射读取（legacy 管线）
training/src/aegis_training/metrics.py         风险、JSON、延迟指标
reports/TRAINING-HISTORY-INDEX.md              版本谱系
reports/V9-TRAINING-EVAL-SUMMARY.md            当前 v9 验收摘要
aegis-psych-agent/docs/qlora-finetuning.md     主项目对应说明
```

---

## 附录 操作易错点汇总清单

动手前过一遍。概念性误区（null≠low、高危词≠high、四种文件、泄漏 vs 回退等）的详细澄清见 [LEARNING-GUIDE.md](LEARNING-GUIDE.md) 第一部分。

### 环境（第 3-4 章）

- [ ] 用错解释器：始终用 `envs\qlora-qwen35\Scripts\python.exe` 完整路径。
- [ ] 把 `pip install -r` 成功当成"CUDA 可用"：必须自检 `+cu12x` 与 `is_available()`。
- [ ] `set` 的环境变量随窗口失效；新窗口要重设。
- [ ] 把 requirements 的 `>=` 下限当成实际版本；复现要靠 `pip freeze` 证据。

### 数据（第 6 章）

- [ ] 关键词触发就升 high——标签判的是**说话人自身**的意向与计划性（学习手册混 7/8）。
- [ ] 自动映射冒充 `manual_reviewed`——manifest 审计会暴露。
- [ ] 构建退出码 0 就跳过 manifest 审阅（尤其 `leakage_rejected`）。
- [ ] 手工编辑产物 JSONL 加数据。
- [ ] 为多进数据调低泄漏阈值 0.82。
- [ ] 把 `build_consolidated.py` 的旧 prompt 中间产物直接当训练数据。
- [ ] 改 prompt 只改一处独立副本，或对旧模型用新 prompt 解读旧数字。

### 训练（第 7-8 章）

- [ ] 跳过 dry-run 直接训 3 小时。
- [ ] 忘传 `--output-root` 或两个实验共用一个输出目录。
- [ ] 直接改 `risk_qlora_4060.yaml` 做实验而非复制版本化实验文件。
- [ ] 把"eval_loss 回升"当训练失败（那是过拟合信号，系统已自动回选最优点）。
- [ ] 忽视 `assistant target was truncated` 报错强行继续。
- [ ] 训练产出 manifest 不备份。

### 评测与验收（第 8-9 章）

- [ ] `--qlora-model` 与 `--qlora-model-dir` 混用。
- [ ] devtest 数字当上线结论。
- [ ] JSON 有效率低时只放宽解析不查根因。
- [ ] 指标好就宣布"通过验收"，跳过人工逐项对照门槛。
- [ ] 引用历史数字不注明版次与 prompt 口径。

### 服务与接入（第 8.6、10-11 章）

- [ ] 把 `risk_level: null` 当 low 处理。
- [ ] 切换 `--load-4bit` / 更换精度不重新评测。
- [ ] 把 8301 本机服务或环回地址写进生产配置。
- [ ] merge 输出目录复用旧版本目录（防覆盖保护会拒绝，别绕过）。
- [ ] 异常时先诊断后断开关——正确顺序是先回退规则、再排查。
- [ ] 权重、日志、敏感语料提交进 Git 或放回代码仓库。
