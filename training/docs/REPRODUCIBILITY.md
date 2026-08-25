# QLoRA 实验可复现与证据记录规范

本文件规定一次 Aegis 风险 QLoRA 训练如何留下足够证据，使另一位维护者能够判断：使用了什么基座、什么数据、什么参数、什么环境、什么代码，以及结果是否真的对应当前 v9。它不替代 [training/README.md](../README.md)，而是定义每次实验的记录要求。

## 1. 可复现边界

AegisTraining 当前可以复现**处理流程和训练脚本**，但不应把“当前本机存在的权重和外部原始数据”误称为干净 checkout 即可复现。以下内容通常位于被忽略目录或外部受控存储：

- 官方 Qwen3.5-2B-Base safetensors 快照；
- HF cache、CUDA 环境、checkpoint、adapter、merged 权重和训练日志；
- PsySUICIDE、suicide、metaphor 等外部原始数据；
- 主项目 checkout 中的冻结 fixture。

因此每次实验必须保存输入来源、revision、hash、授权状态和实际命令。只有代码、配置和报告摘要，没有输入工件证据，不构成完整复现包。

## 2. 实验记录最小字段

建议每次训练建立一个唯一实验 ID，例如：

```text
20260824-v9-r1
```

至少记录以下信息：

```yaml
experiment_id: 20260824-v9-r1
status: research-only # research-only | release-candidate | production-approved | rejected
training_repo_commit: <git rev-parse HEAD>
project_repo_commit: <aegis-psych-agent commit containing fixture>
config_path: training/configs/risk_qlora_4060.yaml
config_sha256: <sha256>
data_manifest_path: data/risk_sft_v9/manifest.json
data_manifest_sha256: <sha256>
prompt_contract: v2
prompt_sha256: <sha256 of exact prompt text>
base_repo: Qwen/Qwen3.5-2B-Base
base_revision: b1485b2fa6dfa1287294f269f5fb618e03d52d7c
base_snapshot_sha256: <manifest or per-shard hashes>
python: <python --version>
pytorch: <torch.__version__>
cuda: <torch.version.cuda>
driver: <nvidia-smi driver version>
transformers: <version>
peft: <version>
bitsandbytes: <version>
gpu: <device name and total memory>
compute_dtype: bfloat16 | float16
seed: 42
train_command: <full command>
merge_command: <full command>
eval_command: <full command>
train_output: <checkpoint directory>
merged_output: <merged directory>
training_manifest: <path>
stress_report: <path>
devtest_report: <path>
release_record: <path>
rollback_version: <version or none>
```

## 3. 环境快照

在隔离环境中执行并把输出保存为实验附件：

```bat
D:\AegisTraining\envs\qlora-qwen35\Scripts\python.exe -c "import sys,torch,transformers,peft,bitsandbytes,datasets,safetensors; print('python=',sys.version); print('torch=',torch.__version__,'cuda=',torch.version.cuda,'available=',torch.cuda.is_available()); print('transformers=',transformers.__version__); print('peft=',peft.__version__); print('bitsandbytes=',getattr(bitsandbytes,'__version__','unknown')); print('datasets=',datasets.__version__); print('safetensors=',safetensors.__version__); print('device=',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

同时保存：

```bat
nvidia-smi
D:\AegisTraining\envs\qlora-qwen35\Scripts\python.exe -m pip freeze
```

注意：`requirements-qlora.txt` 使用下限版本，不等于实际安装版本。真正复现必须保留 `pip freeze`，尤其是 Transformers、PyTorch、PEFT、bitsandbytes 和 CUDA 版本。

## 4. Git 和路径快照

```bat
cd /d D:\AegisTraining
git status --short
git rev-parse HEAD
git diff -- training/configs training/scripts training/src

cd /d D:\PythonProject\aegis-psych-agent
git rev-parse HEAD
git status --short
```

记录以下环境变量的实际值，但不要把密钥写入日志：

```bat
set AEGIS_TRAINING_ROOT
set AEGIS_PROJECT_ROOT
set AEGIS_PROJECT_CORPUS
set HF_HOME
```

`AEGIS_PROJECT_CORPUS` 必须指向已提交的 fixture。训练脚本不会把工作区未提交改动当作正式数据来源；如果使用外部 fixture，必须记录其来源和 hash。

## 5. 输入工件和哈希

### 5.1 基座

`base_model_gate.py` 只能验证本地结构和权重清单，不能证明文件下载来源。实验记录应同时保存：

- Hugging Face repo 和固定 revision；
- 下载时间和下载工具；
- `config.json`、`model.safetensors.index.json` 的 SHA-256；
- 每个 safetensors shard 的 SHA-256，或受控快照 manifest；
- 许可证和使用范围；
- 是否有 Ollama 转换 provenance。

PowerShell 示例：

```powershell
Get-FileHash D:\AegisTraining\models\Qwen3.5-2B-Base\config.json -Algorithm SHA256
Get-FileHash D:\AegisTraining\models\Qwen3.5-2B-Base\model.safetensors.index.json -Algorithm SHA256
Get-ChildItem D:\AegisTraining\models\Qwen3.5-2B-Base -Filter *.safetensors | Get-FileHash -Algorithm SHA256
```

### 5.2 数据

数据构建完成后至少保存：

```bat
certutil -hashfile D:\AegisTraining\data\risk_sft_v9\manifest.json SHA256
certutil -hashfile D:\AegisTraining\data\risk_sft_v9\train.jsonl SHA256
certutil -hashfile D:\AegisTraining\data\risk_sft_v9\dev.jsonl SHA256
certutil -hashfile D:\AegisTraining\data\risk_sft_v9\test.jsonl SHA256
```

manifest 必须被人工检查：

- `schema_version`；
- `seed`；
- system prompt version 和全文；
- sources；
- `relabel_summary`；
- `leakage_rejected` 数量和样本；
- train/dev/devtest 数量与分布；
- hard negatives 和 medium 扩量数量。

不能只保存最终 JSONL 而丢失 manifest；JSONL 没有足够来源信息时不可审计。

### 5.3 配置和 prompt

```bat
certutil -hashfile D:\AegisTraining\training\configs\risk_qlora_4060.yaml SHA256
```

prompt hash 应基于传给 tokenizer 的**完整 system prompt 原文**，不要只记录“v2”这个标签。训练、评测、服务和主项目必须使用同一 prompt contract；旧版模型评测必须绑定旧版 prompt，不能用 v2 重新解释旧版结果。

## 6. 标准命令记录

一次正式实验的命令顺序应是：

```bat
set AEGIS_TRAINING_ROOT=D:\AegisTraining
set AEGIS_PROJECT_ROOT=D:\PythonProject\aegis-psych-agent
set AEGIS_PROJECT_CORPUS=%AEGIS_PROJECT_ROOT%\eval\fixtures\representative_corpus.json

python training\scripts\prepare_risk_sft_v4.py ^
  --consolidated-root "D:\AegisTraining\training\data\consolidated_risk_v1" ^
  --corpus "%AEGIS_PROJECT_CORPUS%" ^
  --output-root "D:\AegisTraining\data\risk_sft_v9"

python training\scripts\train_risk_qlora.py ^
  --config "D:\AegisTraining\training\configs\risk_qlora_4060.yaml" ^
  --data-root "D:\AegisTraining\data\risk_sft_v9" ^
  --snapshot-dir "D:\AegisTraining\models\Qwen3.5-2B-Base" ^
  --output-root "D:\AegisTraining\checkpoints\aegis-risk-qwen3.5-2b-v9" ^
  --dry-run

python training\scripts\train_risk_qlora.py ^
  --config "D:\AegisTraining\training\configs\risk_qlora_4060.yaml" ^
  --data-root "D:\AegisTraining\data\risk_sft_v9" ^
  --snapshot-dir "D:\AegisTraining\models\Qwen3.5-2B-Base" ^
  --output-root "D:\AegisTraining\checkpoints\aegis-risk-qwen3.5-2b-v9"

python training\scripts\merge_risk_qlora.py ^
  --snapshot-dir "D:\AegisTraining\models\Qwen3.5-2B-Base" ^
  --adapter-dir "D:\AegisTraining\checkpoints\aegis-risk-qwen3.5-2b-v9\adapter" ^
  --output-dir "D:\AegisTraining\exports\aegis-risk-qwen3.5-2b-v9-merged"

python training\scripts\eval_risk_qlora.py ^
  --original-model qwen3.5:2b ^
  --qlora-model-dir "D:\AegisTraining\exports\aegis-risk-qwen3.5-2b-v9-merged" ^
  --holdout "%AEGIS_PROJECT_CORPUS%" ^
  --timeout 8 ^
  --max-new-tokens 64 ^
  --output "D:\AegisTraining\reports\risk-qlora-eval-v9.json"

python training\scripts\eval_risk_devtest.py ^
  --model-dir "D:\AegisTraining\exports\aegis-risk-qwen3.5-2b-v9-merged" ^
  --test-jsonl "D:\AegisTraining\data\risk_sft_v9\test.jsonl" ^
  --output "D:\AegisTraining\reports\risk-qlora-devtest-v9.json"
```

每个命令的完整 stdout/stderr 都应保存。不要只保存最后的摘要数字，因为 gate、target modules、解析失败和单条误判需要回溯。

## 7. 结果和验收证据

至少保存：

- dry-run JSON；
- `training-manifest.json`；
- checkpoint 和 adapter 的目录清单；
- `aegis-export-manifest.json`；
- 冻结 stress 评测 JSON 和 raw predictions；
- devtest 评测 JSON；
- 人工门槛检查表；
- v9 markdown 摘要；
- 评测环境和命令；
- 外部审核或审批记录。

当前配置中的 `evaluation` 字段描述验收目标，但训练/评测脚本目前不会自动读取 YAML 并输出统一的 pass/fail gate。文档和人工验收表必须明确这一点：**指标生成是自动的，门槛结论目前需要按报告人工确认**。如果未来增加自动 gate，应保存 gate 版本和代码 commit，避免仅凭配置数字宣称通过。

## 8. 训练 manifest 的改进要求

当前 manifest 已包含 gate、GPU、dtype、target modules、样本数和峰值显存，但独立复现还需要补充：

- training repo commit；
- config path 和 config SHA-256；
- data manifest SHA-256；
- prompt SHA-256；
- Python、PyTorch、CUDA、Transformers、PEFT、bitsandbytes 版本；
- 完整命令行和输出目录；
- effective batch size、optimizer steps；
- 是否 BF16 fallback；
- 训练开始/结束时间；
- resume/checkpoint 选择记录。

在代码完成自动写入前，可以将这些字段放到实验记录 YAML 或报告附件中；不得用手工记忆替代。

## 9. 外部数据的复现要求

`consolidated_risk_v1/build_consolidated.py` 需要训练根目录下的受控数据和项目冻结 fixture。复现者必须获得：

- 每个外部数据源的名称、版本、获取地址、许可证；
- 原始文件 SHA-256 和获取时间；
- 脱敏和授权审查结论；
- `metaphor_corpus_v1.jsonl`、suicide 数据和 PsySUICIDE 的实际来源；
- 项目 `representative_corpus.json` 的 commit/hash；
- 哪些来源是弱监督、哪些经过人工复核；
- `build_consolidated.py` 的 stdout、manifest 和输出 hash。

不能把被忽略的 `external-data/` 目录当成 Git 依赖；干净 clone 必须通过受控数据包或合规数据源重新准备。

## 10. 回滚和证据保存

发现模型门槛失败、提示词漂移、数据泄漏或服务异常时：

1. 立即关闭主项目 `RISK_QLORA_ENABLED`；
2. 恢复上一版已批准 endpoint/model 或纯规则通道；
3. 保留失败模型、配置、manifest、raw predictions 和日志，不覆盖；
4. 在 `reports/TRAINING-HISTORY-INDEX.md` 记录失败原因；
5. 只有重新生成完整证据包并通过门槛后，才能提升为 release candidate。

研究数据和模型工件也必须遵守原始数据的授权和保留期限，不能为了“方便复现”无限期复制敏感语料。
